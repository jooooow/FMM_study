#include <iostream>
#include <random>
#include <iomanip>
#include <omp.h>

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

int main()
{
	const int level = 4;
	const int N = 1000;
	Molecule molecule[N];

	std::default_random_engine reng(std::random_device{}());
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

	int cell_sum = 0;
	int level_offset[level + 1];
	int level_cell_num[level + 1];
	for (int i = 0; i <= level; i++)
	{
		int num = pow(2, i * 2);
		level_offset[i] = cell_sum;
		level_cell_num[i] = num;
		cell_sum += num;
	}
	float* M = new float[cell_sum];
	float* L = new float[cell_sum];
	std::fill_n(M, cell_sum, 0.0f);
	std::fill_n(L, cell_sum, 0.0f);

	//P2M
	int n = pow(2, level);
	int cell_num = n * 2;
	for (int i = 0; i < N; i++)
	{
		Molecule* m = molecule + i;
		int idx_x = m->x * n;
		int idx_y = m->y * n;
		int midx = ComposeMortonIndex({ idx_x,idx_y }, level);
		M[level_offset[level] + midx] += m->q;
	}

	//M2M
	for (int l = level; l > 0; l--)
	{
		int offset = level_offset[l];
		int poffset = level_offset[l - 1];
		int num = level_cell_num[l];
		for (int c = 0; c < num; c++)
		{
			M[poffset + c / 4] += M[offset + c];
		}
	}

	//M2L
	for (int l = 2; l <= level; l++)
	{
		int num = level_cell_num[l];
		int offset = level_offset[l];
		for (int c = 0; c < num; c++)
		{
			IndexDim2 idx = DecomposeMortonIndex(c, l);
			IndexDim2 pidx = DecomposeMortonIndex(c / 4, l - 1);
			for (int pnidx_y = pidx.y - 1; pnidx_y <= pidx.y + 1; pnidx_y++)
			{
				for (int pnidx_x = pidx.x - 1; pnidx_x <= pidx.x + 1; pnidx_x++)
				{
					if (pnidx_y < 0 || pnidx_y > pow(2, l - 1) - 1 || pnidx_x < 0 || pnidx_x > pow(2, l - 1) - 1)
						continue;
					if (pnidx_y == pidx.y && pnidx_x == pidx.x)
						continue;
					int pmidx = ComposeMortonIndex({ pnidx_x ,pnidx_y }, l - 1);
					for (int i = 0; i < 4; i++)
					{
						int pcmidx = pmidx * 4 + i;
						IndexDim2 pcidx = DecomposeMortonIndex(pcmidx, l);
						if (!(abs(idx.x - pcidx.x) <= 1 && abs(idx.y - pcidx.y) <= 1))
						{
							float distance = sqrt(pow(abs(idx.x - pcidx.x) / pow(2, l), 2) + pow(abs(idx.y - pcidx.y) / pow(2, l), 2));
							L[offset + c] += M[offset + pcmidx] / distance;
						}
					}
				}
			}
		}
	}

	//L2L
	for (int l = 3; l <= level; l++)
	{
		int num = level_cell_num[l];
		int offset = level_offset[l];
		int poffset = level_offset[l - 1];
		for (int c = 0; c < num; c++)
		{
			L[offset + c] += L[poffset + c / 4];
		}
	}

	//L2P
	int offset = level_offset[level];
	for (int i = 0; i < N; i++)
	{
		Molecule* m = molecule + i;
		int idx_x = m->x * n;
		int idx_y = m->y * n;
		int midx = ComposeMortonIndex({ idx_x,idx_y }, level);
		m->u_appx += L[offset + midx];
	}

	//P2P
	for (int i = 0; i < N; i++)
	{
		Molecule* tm = molecule + i;
		int x1 = tm->x * n;
		int y1 = tm->y * n;
		for (int j = 0; j < N; j++)
		{
			Molecule* sm = molecule + j;
			int x2 = sm->x * n;
			int y2 = sm->y * n;
			if (abs(x1 - x2) <= 1 && abs(y1 - y2) <= 1)
			{
				float distance = sqrt(pow(tm->x - sm->x, 2) + pow(tm->y - sm->y, 2));
				if (distance > 0)
					tm->u_appx += sm->q / distance;
			}
		}
	}

	for (int i = 0; i < N; i++)
	{
		std::cout << molecule[i] << std::endl;
	}

	delete[] M;
	delete[] L;

	return 0;
}