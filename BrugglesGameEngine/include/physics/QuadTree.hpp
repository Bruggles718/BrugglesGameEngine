#pragma once

#include <memory>
#include <vector>
#include "physics/CollisionObject.hpp"

namespace bruggles {
	namespace physics {
		struct QuadTree {
			float x = 0;
			float y = 0;
			float w = 0;
			float h = 0;

			int depth = 4;

			std::unique_ptr<QuadTree> ne;
			std::unique_ptr<QuadTree> nw;
			std::unique_ptr<QuadTree> se;
			std::unique_ptr<QuadTree> sw;

			int capacity = 3;
			bool divided = false;

			std::vector<CollisionObject*> bodies;

			QuadTree(float i_x, float i_y, float i_w, float i_h, int i_depth);

			bool Contains(CollisionObject* body, std::unordered_map<Uint64, std::pair<Vector2, Vector2>>& tlbrs);

			bool Insert(CollisionObject* body, std::unordered_map<Uint64, std::pair<Vector2, Vector2>>& tlbrs);

			void Divide();

			std::vector<std::pair<CollisionObject*, CollisionObject*>> GetSweepAndPrunePairs(std::unordered_map<Uint64, std::pair<Vector2, Vector2>>& tlbrs);
		};
	}
}