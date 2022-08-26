#include <iostream>
#include <random>
#include <iomanip>
#include <omp.h>
#include <algorithm>
#include <numeric>
#include <map>

struct IndexDim2
{
	int x;
	int y;
};

int ComposeMortonIndex(const IndexDim2& idx, int level)
{
	int index = 0;
	IndexDim2 raw_idx = idx;
	for (int l = 0; l < level; l++)
	{
		index += ((raw_idx.x & 0x01) << (2 * l + 1)) + ((raw_idx.y & 0x01) << (2 * l));
		raw_idx.x >>= 1;
		raw_idx.y >>= 1;
	}
	return index;
}

IndexDim2 DecomposeMortonIndex(const int& idx, int level)
{
	int raw_indx = idx;
	IndexDim2 res_idx = { 0,0 };
	for (int l = 0; l < level; l++)
	{
		res_idx.y += (raw_indx & 0x01) << l;
		raw_indx >>= 1;
		res_idx.x += (raw_indx & 0x01) << l;
		raw_indx >>= 1;
	}
	return res_idx;
}

struct Molecule
{
	float x;
	float y;
	float q;
	float u_appx;
	float u_real;
	friend std::ostream& operator << (std::ostream& os, const Molecule& m);
	static float Distance(const Molecule* a, const Molecule* b)
	{
		return sqrt(powf(a->x - b->x, 2) + powf(a->y - b->y, 2));
	}
};

struct MoleculeInfo
{
	int idx;
	int cell_idx;
};

std::ostream& operator << (std::ostream& os, const Molecule& m)
{
	os << std::fixed << std::setprecision(4) << "molecule(" << m.x << "," << m.y << "," << m.y << "\t" << m.q << "\t" << m.u_real << "," << m.u_appx << ")";
	return os;
}

void GetReal(Molecule* m, const int N)
{
	#pragma omp parallel for
	for (int t = 0; t < N; t++)
	{
		Molecule* mt = m + t;
		float u_sum = 0.0f;
		for (int s = 0; s < N; s++)
		{
			Molecule* sm = m + s;
			float distance = Molecule::Distance(mt, sm);
			if (distance > 0)
				u_sum += sm->q / distance;
		}
		mt->u_real = u_sum;
	}
}

struct Cell
{
	Cell() { M = 0.0f; L = 0.0f; midx = 0; }
	float M;
	float L;
	int midx;
};

struct Level
{
	Level(std::vector<int>& midxs)
	{
		num = midxs.size();
		cells.resize(num);
		for (int i = 0; i < num; i++)
		{
			cells.at(i).midx = midxs.at(i);
			midx2order.insert(std::make_pair(midxs.at(i), i));
		}
	}
	Cell& GetCellByOrder(int order)
	{
		return cells.at(order);
	}
	Cell& GetCellByMorton(int midx)
	{
		return GetCellByOrder(midx2order[midx]);
	}
	int GetOrderByMorton(int midx)
	{
		return midx2order[midx];
	}
	int CheckExistenceByMorton(int midx)
	{
		return midx2order.count(midx);
	}
	int num;
	std::vector<Cell> cells;
	std::map<int, int> midx2order;
};

class Grid
{
public:
	Grid(int _max_level, int _N, Molecule* _ms) : max_level(_max_level), N(_N), ms(_ms), total_cell_number(0)
	{
		cell_offset_eachlevel = new int[max_level + 1]{ 0 };
		sorted_ms_info = new MoleculeInfo[N];
		for (int i = 0; i < N; i++)
		{
			int x = ms[i].x * pow(2, max_level);
			int y = ms[i].y * pow(2, max_level);
			int midx = ComposeMortonIndex({ x,y }, max_level);
			sorted_ms_info[i] = { i,midx };
		}
		std::sort(sorted_ms_info, sorted_ms_info + N, [](MoleculeInfo a, MoleculeInfo b) {return a.cell_idx < b.cell_idx; });

		for (int l = 0; l <= max_level; l++)
		{
			std::vector<int> morton_indices;
			for (int i = 0; i < N; i++)
			{
				if (i == 0 || static_cast<int>(sorted_ms_info[i].cell_idx / pow(4, max_level - l)) != static_cast<int>(sorted_ms_info[i - 1].cell_idx / pow(4, max_level - l)))
				{
					morton_indices.push_back(sorted_ms_info[i].cell_idx / pow(4, max_level - l));
				}
			}
			Level* level = new Level(morton_indices);
			levels.push_back(level);
			if (l == max_level)
			{
				molecule_offsets = new int[morton_indices.size()];
			}
		}

		int count = 0;
		for (int i = 0; i < N; i++)
		{
			if (i == 0 || static_cast<int>(sorted_ms_info[i].cell_idx) != static_cast<int>(sorted_ms_info[i - 1].cell_idx))
			{
				molecule_offsets[count] = i;
				count++;
			}
		}

	}
	~Grid()
	{
		delete[] cell_offset_eachlevel;
		delete[] sorted_ms_info;
		for (int l = 0; l <= max_level; l++)
		{
			Level* level = levels.at(l);
			delete level;
		}
		delete[] molecule_offsets;
	}

	void P2M()
	{
		Level* leaf_level = levels.at(max_level);
		for (int i = 0; i < leaf_level->num; i++)
		{
			Cell& cell = leaf_level->cells.at(i);
			int end = i == leaf_level->num - 1 ? N : molecule_offsets[i + 1];
			for (int j = molecule_offsets[i]; j < end; j++)
			{
				cell.M += ms[sorted_ms_info[j].idx].q;
			}
		}
	}

	void M2M()
	{
		for (int l = max_level; l > 0; l--)
		{
			Level* this_level = levels.at(l);
			Level* parent_level = levels.at(l - 1);
			for (int i = 0; i < this_level->num; i++)
			{
				Cell& this_cell = this_level->cells.at(i);
				Cell& parent_cell = parent_level->GetCellByMorton(this_cell.midx / 4);
				parent_cell.M += this_cell.M;
			}
		}
	}

	void M2L()
	{
		for (int l = 2; l <= max_level; l++)
		{
			Level* this_level = levels.at(l);
			Level* parent_level = levels.at(l - 1);
			for (int i = 0; i < this_level->num; i++)
			{
				Cell& this_cell = this_level->cells.at(i);
				Cell& parent_cell = parent_level->GetCellByMorton(this_cell.midx / 4);
				IndexDim2 this_cord = DecomposeMortonIndex(this_cell.midx, l);
				IndexDim2 parent_cord = DecomposeMortonIndex(parent_cell.midx, l - 1);
				for (int y = parent_cord.y - 1; y <= parent_cord.y + 1; y++)
				{
					for (int x = parent_cord.x - 1; x <= parent_cord.x + 1; x++)
					{
						if (x < 0 || x > pow(2, l - 1) - 1 || y < 0 || y > pow(2, l - 1) - 1)
							continue;
						if (x == parent_cord.x && y == parent_cord.y)
							continue;
						int parent_neighbour_midx = ComposeMortonIndex({ x,y }, l - 1);
						if (parent_level->CheckExistenceByMorton(parent_neighbour_midx))
						{
							for (int z = 0; z < 4; z++)
							{
								if (this_level->CheckExistenceByMorton(parent_neighbour_midx * 4 + z))
								{
									Cell& cousin_cell = this_level->GetCellByMorton(parent_neighbour_midx * 4 + z);
									IndexDim2 cousin_cord = DecomposeMortonIndex(cousin_cell.midx, l);
									if (!(abs(this_cord.x - cousin_cord.x) <= 1 && abs(this_cord.y - cousin_cord.y) <= 1))
									{
										float distance = sqrt(pow(abs(this_cord.x - cousin_cord.x) / pow(2, l), 2) + pow(abs(this_cord.y - cousin_cord.y) / pow(2, l), 2));
										this_cell.L += cousin_cell.M / distance;
									}
								}
							}
						}
					}
				}
			}
		}
	}

	void L2L()
	{
		for (int l = 3; l <= max_level; l++)
		{
			Level* this_level = levels.at(l);
			Level* parent_level = levels.at(l - 1);
			for (int i = 0; i < this_level->num; i++)
			{
				Cell& this_cell = this_level->cells.at(i);
				Cell& parent_cell = parent_level->GetCellByMorton(this_cell.midx / 4);
				this_cell.L += parent_cell.L;
			}
		}
	}

	void L2P()
	{
		for (int i = 0; i < N; i++)
		{
			Molecule* m = ms + sorted_ms_info[i].idx;
			Cell& cell = levels.at(max_level)->GetCellByMorton(sorted_ms_info[i].cell_idx);
			m->u_appx += cell.L;
		}
	}

	void P2P()
	{
		Level* leaf_level = levels.at(max_level);
		for (int p = 0; p < leaf_level->num; p++)
		{
			Cell& cell = leaf_level->cells.at(p);
			IndexDim2 cord = DecomposeMortonIndex(cell.midx, max_level);
			for (int y = cord.y - 1; y <= cord.y + 1; y++)
			{
				for (int x = cord.x - 1; x <= cord.x + 1; x++)
				{
					if (x < 0 || x > pow(2, max_level) - 1 || y < 0 || y > pow(2, max_level) - 1)
						continue;
					int brother_midx = ComposeMortonIndex({ x,y }, max_level);
					if (leaf_level->CheckExistenceByMorton(brother_midx))
					{
						int bi = leaf_level->GetOrderByMorton(brother_midx);
						int end1 = p == leaf_level->num - 1 ? N : molecule_offsets[p + 1];
						for (int i = molecule_offsets[p]; i < end1; i++)
						{
							Molecule* mi = ms + sorted_ms_info[i].idx;
							int end2 = bi == leaf_level->num - 1 ? N : molecule_offsets[bi + 1];
							for (int j = molecule_offsets[bi]; j < end2; j++)
							{
								Molecule* mj = ms + sorted_ms_info[j].idx;
								float distance = sqrt(pow(mi->x - mj->x, 2) + pow(mi->y - mj->y, 2));
								if (distance > 0)
									mi->u_appx += mj->q / distance;
							}
							
						}
					}
				}
			}
		}
	}

private:
	int max_level;
	int N;
	Molecule* ms;
	MoleculeInfo* sorted_ms_info;
	int total_cell_number;
	int* cell_offset_eachlevel;
	int* molecule_offsets;
	std::vector<Level*> levels;
};

int main()
{
	const int level = 4;
	const int N = 100;
	Molecule molecule[N];

	std::default_random_engine reng(std::random_device{}());
	//reng.seed(1);
	std::uniform_real_distribution<float> rd(0.0, 1.0);
	for (int i = 0; i < N; i++)
	{
		molecule[i].x = rd(reng);
		molecule[i].y = rd(reng);
		molecule[i].q = rd(reng);
		molecule[i].u_appx = 0.0f;
		molecule[i].u_real = 0.0f;
	}
	GetReal(molecule, N);

	Grid grid(level, N, molecule);
	grid.P2M();
	grid.M2M();
	grid.M2L();
	grid.L2L();
	grid.L2P();
	grid.P2P();

	for (int i = 0; i < N; i++)
	{
		std::cout << molecule[i] << std::endl;
	}

	return 0;
}