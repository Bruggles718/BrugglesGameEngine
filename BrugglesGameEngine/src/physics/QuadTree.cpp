#include "physics/QuadTree.hpp"
#include "physics/EndPoint.hpp"
#include <iostream>

namespace bruggles {
	namespace physics {
		QuadTree::QuadTree(float i_x, float i_y, float i_w, float i_h, int i_depth) {
			this->x = i_x;
			this->y = i_y;
			this->w = i_w;
			this->h = i_h;
			this->depth = i_depth;
		}

		bool QuadTree::Contains(CollisionObject* body, std::unordered_map<Uint64, std::pair<Vector2, Vector2>>& tlbrs) {
			if (!body->collider) {
				return false;
			}
			std::pair<Vector2, Vector2> tlbr = tlbrs[body->m_uniqueID];
			float top = tlbr.first.y;
			float left = tlbr.first.x;
			float bottom = tlbr.second.y;
			float right = tlbr.second.x;
			return (
				this->x < right &&
				this->x + this->w > left &&
				this->y < bottom &&
				this->y + this->h > top
			);
		}

		bool QuadTree::Insert(CollisionObject* body, std::unordered_map<Uint64, std::pair<Vector2, Vector2>>& tlbrs) {
			if (!this->Contains(body, tlbrs)) {
				return false;
			}

			if (!this->divided && this->bodies.size() < this->capacity || depth == 0) {
				this->bodies.push_back(body);
				return true;
			}
			else {
				if (!this->divided) {
					this->Divide();
					this->divided = true;
					for (int i = 0; i < this->bodies.size(); i += 1) {
						this->ne->Insert(this->bodies.at(i), tlbrs);
						this->nw->Insert(this->bodies.at(i), tlbrs);
						this->se->Insert(this->bodies.at(i), tlbrs);
						this->sw->Insert(this->bodies.at(i), tlbrs);
					}
				}
				this->bodies.clear();
			}
			bool inserted = false;
			if (this->ne != nullptr) {
				if (this->ne->Insert(body, tlbrs)) {
					inserted = true;
				}
			}
			if (this->nw != nullptr) {
				if (this->nw->Insert(body, tlbrs)) {
					inserted = true;
				}
			}
			if (this->se != nullptr) {
				if (this->se->Insert(body, tlbrs)) {
					inserted = true;
				}
			}
			if (this->sw != nullptr) {
				if (this->sw->Insert(body, tlbrs)) {
					inserted = true;
				}
			}
			return inserted;
		}

		void QuadTree::Divide() {
			if (!this->divided) {
				this->nw = std::make_unique<QuadTree>(x, y, w / 2, h / 2, depth-1);
				this->ne = std::make_unique<QuadTree>(x + w / 2, y, w / 2, h / 2, depth - 1);
				this->sw = std::make_unique<QuadTree>(x, y + h / 2, w / 2, h / 2, depth - 1);
				this->se = std::make_unique<QuadTree>(x + w / 2, y + h / 2, w / 2, h / 2, depth - 1);
			}
		}

		std::vector<std::pair<CollisionObject*, CollisionObject*>> QuadTree::GetSweepAndPrunePairs(std::unordered_map<Uint64, std::pair<Vector2, Vector2>>& tlbrs) {
			if (this->divided) {
				std::vector<std::pair<CollisionObject*, CollisionObject*>> result{};
				std::vector<std::pair<CollisionObject*, CollisionObject*>> nwPairs = nw->GetSweepAndPrunePairs(tlbrs);
				std::vector<std::pair<CollisionObject*, CollisionObject*>> nePairs = ne->GetSweepAndPrunePairs(tlbrs);
				std::vector<std::pair<CollisionObject*, CollisionObject*>> swPairs = sw->GetSweepAndPrunePairs(tlbrs);
				std::vector<std::pair<CollisionObject*, CollisionObject*>> sePairs = se->GetSweepAndPrunePairs(tlbrs);
				result.insert(result.end(), std::make_move_iterator(nwPairs.begin()), std::make_move_iterator(nwPairs.end()));
				result.insert(result.end(), std::make_move_iterator(nePairs.begin()), std::make_move_iterator(nePairs.end()));
				result.insert(result.end(), std::make_move_iterator(swPairs.begin()), std::make_move_iterator(swPairs.end()));
				result.insert(result.end(), std::make_move_iterator(sePairs.begin()), std::make_move_iterator(sePairs.end()));
				return result;
			}

			std::vector<EndPoint> xPoints{};
			std::vector<EndPoint> yPoints{};

			for (CollisionObject* object : bodies) {
				if (!object->collider) {
					continue;
				}

				std::pair<Vector2, Vector2> topLeftBottomRight = tlbrs[object->m_uniqueID];

				// Get x min and max
				// Insert into list
				// Get y min and max
				// Insert into second list

				EndPoint minX{
					object,
					object->m_uniqueID,
					topLeftBottomRight.first.x,
					true
				};
				EndPoint maxX{
					object,
					object->m_uniqueID,
					topLeftBottomRight.second.x,
					false
				};
				EndPoint minY{
					object,
					object->m_uniqueID,
					topLeftBottomRight.first.y,
					true
				};
				EndPoint maxY{
					object,
					object->m_uniqueID,
					topLeftBottomRight.second.y,
					false
				};

				//std::cout << object->m_uniqueID << " x bounds: " << minX.value << " " << maxX.value << std::endl;
				//std::cout << object->m_uniqueID << " y bounds: " << minY.value << " " << maxY.value << std::endl;

				BinaryInsert(xPoints, minX);
				BinaryInsert(xPoints, maxX);
				BinaryInsert(yPoints, minY);
				BinaryInsert(yPoints, maxY);
			}

			std::unordered_map<Uint64, std::vector<EndPoint>> xPairs;
			std::unordered_map<Uint64, std::vector<EndPoint>> yPairs;

			std::vector<Uint64> insideListX{};

			for (int i = 0; i < xPoints.size(); i++) {
				EndPoint& xPoint = xPoints[i];
				if (xPoint.isMin) {
					for (Uint64 insideID : insideListX) {
						if (insideID != xPoint.id) {
							xPairs[insideID].push_back(xPoint);
						}
					}
					insideListX.push_back(xPoint.id);

				}
				else {
					auto itr = std::find(insideListX.begin(), insideListX.end(), xPoint.id);
					insideListX.erase(itr);
				}
			}

			for (EndPoint& e : xPoints) {
				auto& pairsWithE = xPairs[e.id];
				for (EndPoint& pairWithE : pairsWithE) {
					xPairs[pairWithE.id].push_back(e);
				}
			}

			std::vector<Uint64> insideListY{};

			for (int i = 0; i < yPoints.size(); i++) {
				EndPoint& yPoint = yPoints[i];
				if (yPoint.isMin) {
					for (Uint64 insideID : insideListY) {
						if (insideID != yPoint.id) {
							yPairs[insideID].push_back(yPoint);
						}
					}
					insideListY.push_back(yPoint.id);
				}
				else {
					auto itr = std::find(insideListY.begin(), insideListY.end(), yPoint.id);
					insideListY.erase(itr);
				}
			}

			std::vector<std::pair<CollisionObject*, CollisionObject*>> result{};
			for (CollisionObject* object : bodies) {
				auto& xPointsPotential = xPairs[object->m_uniqueID];
				auto& yPointsPotential = yPairs[object->m_uniqueID];
				std::vector<EndPoint> pairs{};
				std::vector<Uint64> pairIDs{};
				for (auto& xE : xPointsPotential) {
					for (auto& yE : yPointsPotential) {
						if (xE.id == yE.id) {
							pairs.push_back(xE);
							break;
						}
					}
					//pairs.push_back(xE.object);
				}
				for (auto& pairObject : pairs) {
					if (std::find(pairIDs.begin(), pairIDs.end(), pairObject.id) == pairIDs.end()) {
						result.emplace_back(object, pairObject.object);
						pairIDs.push_back(pairObject.id);
					}
				}
			}

			// std::cout << "result size: " << result.size() << std::endl;

			return result;
		}
	}
}