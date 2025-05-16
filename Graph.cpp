// single_scene_graph.cpp

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <functional>
#include <array>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtc/type_ptr.hpp>

using glm::vec3;
using glm::vec4;
using glm::mat4;
using glm::quat;

// ---------------------------------------------
// Physics preparation (bounding boxes)

struct BoundingBox {
    vec3 min, max;
};

// ---------------------------------------------
// Transform class

class Transform {
private:
    vec3 position;
    quat rotation;
    vec3 scale;
    mutable mat4 localMatrix;
    mutable bool dirty;
public:
    Transform()
        : position(0.0f), rotation(quat()), scale(1.0f),
          localMatrix(1.0f), dirty(true) {}

    void setPosition(const vec3& pos) { position = pos; dirty = true; }
    void setRotation(const quat& rot)  { rotation = rot; dirty = true; }
    void setScale(const vec3& scl)     { scale    = scl; dirty = true; }

    vec3 getPosition() const { return position; }
    quat getRotation() const { return rotation; }
    vec3 getScale()    const { return scale; }

    mat4 getMatrix() const {
        if (dirty) {
            mat4 t = glm::translate(mat4(1.0f), position);
            mat4 r = glm::mat4_cast(rotation);
            mat4 s = glm::scale(mat4(1.0f), scale);
            localMatrix = t * r * s;
            dirty = false;
        }
        return localMatrix;
    }
};

// ---------------------------------------------
// Forward declaration

class SceneNode;
using SceneNodePtr = std::shared_ptr<SceneNode>;

// ---------------------------------------------
// Partitioning interface

class PartitioningStrategy {
public:
    virtual void insert(const SceneNodePtr& node) = 0;
    virtual void clear() = 0;
    virtual ~PartitioningStrategy() = default;
};

// ---------------------------------------------
// Level of Detail (LOD)

struct LODLevel {
    float distanceThreshold;
    std::string meshName;
};

class LOD {
public:
    std::vector<LODLevel> levels;

    void addLevel(float distance, const std::string& mesh) {
        levels.push_back({distance, mesh});
        std::sort(levels.begin(), levels.end(),
            [](auto& a, auto& b){ return a.distanceThreshold < b.distanceThreshold; });
    }

    std::string getMesh(float distance) const {
        for (auto& lvl : levels)
            if (distance < lvl.distanceThreshold) return lvl.meshName;
        return levels.empty() ? std::string() : levels.back().meshName;
    }
};

// ---------------------------------------------
// Scene Node

class SceneNode : public std::enable_shared_from_this<SceneNode> {
public:
    std::string name;
    Transform transform;
    std::weak_ptr<SceneNode> parent;
    std::vector<SceneNodePtr> children;
    BoundingBox boundingBox;
    LOD lod;
    bool visible;

    SceneNode(const std::string& n)
        : name(n), boundingBox({{-1,-1,-1},{1,1,1}}), visible(true) {}

    void addChild(const SceneNodePtr& child) {
        child->parent = shared_from_this();
        children.push_back(child);
    }
    void removeChild(const SceneNodePtr& child) {
        children.erase(
            std::remove(children.begin(), children.end(), child),
            children.end()
        );
    }

    mat4 getWorldMatrix() const {
        if (auto p = parent.lock())
            return p->getWorldMatrix() * transform.getMatrix();
        return transform.getMatrix();
    }

    void updateWorldMatrix(bool force = false) {
        (void)force;
        for (auto& c : children) c->updateWorldMatrix(force);
    }

    void draw(int depth = 0) const {
        for (int i = 0; i < depth; ++i) std::cout << "  ";
        std::cout << name << " [Visible: " << visible << "]\n";
        for (auto& c : children) c->draw(depth + 1);
    }
};

// ---------------------------------------------
// Octree partitioning

class Octree : public PartitioningStrategy {
    vec3 center;
    float halfSize;
    int depth;
    std::vector<SceneNodePtr> objects;
    std::array<std::unique_ptr<Octree>, 8> children;
public:
    Octree(const vec3& c, float hs, int d = 0)
        : center(c), halfSize(hs), depth(d) {}

    void insert(const SceneNodePtr& node) override {
        objects.push_back(node);
    }

    void clear() override {
        objects.clear();
        for (auto& ch : children) ch.reset();
    }

    void subdivide() {
        float h = halfSize * 0.5f;
        for (int i = 0; i < 8; ++i) {
            vec3 off(
                (i & 1 ?  h : -h),
                (i & 2 ?  h : -h),
                (i & 4 ?  h : -h)
            );
            children[i] = std::make_unique<Octree>(center + off, h, depth + 1);
        }
    }
};

// ---------------------------------------------
// BSP Tree partitioning

class BSPTree : public PartitioningStrategy {
    vec3 normal;
    float distance;
    std::vector<SceneNodePtr> frontList, backList;
    std::unique_ptr<BSPTree> front, back;
public:
    BSPTree(const vec3& n, float d)
        : normal(n), distance(d) {}

    void insert(const SceneNodePtr& node) override {
        frontList.push_back(node);
    }

    void clear() override {
        frontList.clear();
        backList.clear();
        front.reset();
        back.reset();
    }
};

// ---------------------------------------------
// Frustum Culling

class FrustumCuller {
    std::array<vec4,6> planes;

    void extractPlanes(const mat4& m) {
        planes[0] = glm::row(m,3) + glm::row(m,0);
        planes[1] = glm::row(m,3) - glm::row(m,0);
        planes[2] = glm::row(m,3) + glm::row(m,1);
        planes[3] = glm::row(m,3) - glm::row(m,1);
        planes[4] = glm::row(m,3) + glm::row(m,2);
        planes[5] = glm::row(m,3) - glm::row(m,2);
        for (auto& p : planes) {
            float l = glm::length(glm::vec3(p));
            p /= l;
        }
    }

public:
    FrustumCuller(const mat4& projView) {
        extractPlanes(projView);
    }

    bool isVisible(const SceneNodePtr& node) const {
        mat4 wm = node->getWorldMatrix();
        auto bb = node->boundingBox;
        vec3 pts[8] = {
            {bb.min.x,bb.min.y,bb.min.z},{bb.max.x,bb.min.y,bb.min.z},
            {bb.min.x,bb.max.y,bb.min.z},{bb.max.x,bb.max.y,bb.min.z},
            {bb.min.x,bb.min.y,bb.max.z},{bb.max.x,bb.min.y,bb.max.z},
            {bb.min.x,bb.max.y,bb.max.z},{bb.max.x,bb.max.y,bb.max.z}
        };
        for (auto& plane : planes) {
            int out = 0;
            for (auto& p : pts) {
                vec4 wp = wm * vec4(p,1.0f);
                if (glm::dot(glm::vec3(plane), glm::vec3(wp)) + plane.w < 0) out++;
            }
            if (out == 8) { 
                node->visible = false;
                return false;
            }
        }
        node->visible = true;
        return true;
    }
};

// ---------------------------------------------
// Serialization / Deserialization

class Serializer {
    static void serializeNode(const SceneNodePtr& node, std::ostream& os, int indent = 0) {
        std::string ind(indent, ' ');
        os << ind << "Node " << node->name << "\n";

        auto pos = node->transform.getPosition();
        auto rot = node->transform.getRotation();
        auto scl = node->transform.getScale();
        os << ind << "  Position " << pos.x << " " << pos.y << " " << pos.z << "\n";
        os << ind << "  Rotation " << rot.x << " " << rot.y << " " << rot.z << " " << rot.w << "\n";
        os << ind << "  Scale "    << scl.x << " " << scl.y << " " << scl.z << "\n";

        os << ind << "  LODLevels " << node->lod.levels.size() << "\n";
        for (auto& lvl : node->lod.levels)
            os << ind << "    " << lvl.distanceThreshold << " " << lvl.meshName << "\n";

        os << ind << "  BoundingBox "
           << node->boundingBox.min.x << " " << node->boundingBox.min.y << " " << node->boundingBox.min.z << " "
           << node->boundingBox.max.x << " " << node->boundingBox.max.y << " " << node->boundingBox.max.z << "\n";

        os << ind << "  Children " << node->children.size() << "\n";
        for (auto& c : node->children)
            serializeNode(c, os, indent + 4);
    }

public:
    static void serialize(const SceneNodePtr& root, const std::string& filename) {
        std::ofstream ofs(filename);
        serializeNode(root, ofs);
    }

    static SceneNodePtr deserialize(const std::string& filename) {
        std::ifstream ifs(filename);
        std::function<SceneNodePtr(std::istream&)> deser = [&](std::istream& is)->SceneNodePtr {
            std::string tok;
            if (!(is >> tok) || tok != "Node") return nullptr;
            std::string name; is >> name;
            auto node = std::make_shared<SceneNode>(name);

            is >> tok; float px,py,pz; is >> px >> py >> pz;
            node->transform.setPosition({px,py,pz});
            is >> tok; float rx,ry,rz,rw; is >> rx >> ry >> rz >> rw;
            node->transform.setRotation({rw,rx,ry,rz});
            is >> tok; float sx,sy,sz; is >> sx >> sy >> sz;
            node->transform.setScale({sx,sy,sz});

            is >> tok; int lodCount; is >> lodCount;
            for (int i = 0; i < lodCount; ++i) {
                float d; std::string m; is >> d >> m;
                node->lod.addLevel(d,m);
            }

            is >> tok; float minx,miny,minz,maxx,maxy,maxz;
            is >> minx >> miny >> minz >> maxx >> maxy >> maxz;
            node->boundingBox.min = {minx,miny,minz};
            node->boundingBox.max = {maxx,maxy,maxz};

            is >> tok; int childCount; is >> childCount;
            for (int i = 0; i < childCount; ++i) {
                auto c = deser(is);
                if (c) node->addChild(c);
            }

            return node;
        };
        return ifs ? deser(ifs) : nullptr;
    }
};

// ---------------------------------------------
// Minimal CLI UI

class UI {
    SceneNodePtr root;
    std::unique_ptr<PartitioningStrategy> partitioner;
public:
    UI()
        : root(std::make_shared<SceneNode>("Root")),
          partitioner(std::make_unique<Octree>(vec3(0.0f), 100.0f)) {}

    void run() {
        int choice = 0;
        while (choice != 9) {
            std::cout
                << "1.Add Node\n"
                << "2.Remove Node\n"
                << "3.Move Node\n"
                << "4.Print Scene Graph\n"
                << "5.Serialize\n"
                << "6.Deserialize\n"
                << "7.Switch Partitioner\n"
                << "8.Cull & Print Visible\n"
                << "9.Quit\n"
                << "Choice: ";
            std::cin >> choice;
            switch (choice) {
                case 1: addNode();         break;
                case 2: removeNode();      break;
                case 3: moveNode();        break;
                case 4: printGraph(root,0);break;
                case 5: serializeScene();  break;
                case 6: deserializeScene();break;
                case 7: switchPartitioner();break;
                case 8: cullAndPrint();    break;
            }
        }
    }

private:
    void addNode() {
        std::string parentName, nodeName;
        std::cout << "Parent Name: "; std::cin >> parentName;
        auto parent = findNode(parentName, root);
        if (!parent) { std::cout << "Parent not found\n"; return; }
        std::cout << "Node Name: ";  std::cin >> nodeName;
        parent->addChild(std::make_shared<SceneNode>(nodeName));
    }

    void removeNode() {
        std::string name;
        std::cout << "Node Name: "; std::cin >> name;
        auto node = findNode(name, root);
        if (node && node->parent.lock())
            node->parent.lock()->removeChild(node);
    }

    void moveNode() {
        std::string name;
        float x,y,z;
        std::cout << "Node Name: ";        std::cin >> name;
        auto node = findNode(name, root);
        if (!node) return;
        std::cout << "New Position x y z: "; std::cin >> x >> y >> z;
        node->transform.setPosition({x,y,z});
    }

    void printGraph(const SceneNodePtr& node, int depth) {
        node->draw(depth);
    }

    void serializeScene() {
        std::string filename;
        std::cout << "Filename: "; std::cin >> filename;
        Serializer::serialize(root, filename);
    }

    void deserializeScene() {
        std::string filename;
        std::cout << "Filename: "; std::cin >> filename;
        auto newRoot = Serializer::deserialize(filename);
        if (newRoot) root = newRoot;
    }

    void switchPartitioner() {
        int c;
        std::cout << "1.Octree 2.BSP: ";
        std::cin >> c;
        if (c == 1)
            partitioner = std::make_unique<Octree>(vec3(0.0f), 100.0f);
        else
            partitioner = std::make_unique<BSPTree>(vec3(0,1,0), 0.0f);
    }

    void cullAndPrint() {
        partitioner->clear();
        std::vector<SceneNodePtr> all;
        std::function<void(SceneNodePtr)> gather = [&](SceneNodePtr n){
            all.push_back(n);
            for (auto& c : n->children) gather(c);
        };
        gather(root);

        for (auto& n : all)
            partitioner->insert(n);

        FrustumCuller culler(mat4(1.0f));
        std::cout << "Visible Nodes:\n";
        for (auto& n : all)
            if (culler.isVisible(n))
                std::cout << "  " << n->name << "\n";
    }

    SceneNodePtr findNode(const std::string& name, const SceneNodePtr& node) {
        if (node->name == name) return node;
        for (auto& c : node->children)
            if (auto r = findNode(name, c)) return r;
        return nullptr;
    }
};

// ---------------------------------------------
// Main entry point

int main() {
    UI ui;
    ui.run();
    return 0;
}