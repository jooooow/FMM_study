#include <iostream>
#include <iomanip>
#include <random>

#define N 16
#define D 4

struct Molecule
{
	float x;
	float y;
	float z;
	float q;
	float u_appx;
	float u_real;
	friend std::ostream& operator << (std::ostream& os, const Molecule& m);
	static float Distance(const Molecule* a, const Molecule* b)
	{
		return sqrt(powf(a->x - b->x, 2) + powf(a->y - b->y, 2) + powf(a->z - b->z, 2));
	}
};

std::ostream& operator << (std::ostream& os, const Molecule& m)
{
	os << std::fixed << std::setprecision(5) << "(" << m.x << "," << m.y << "," << m.z << ")\t" << m.q << "\t" << m.u_appx << "\t" << m.u_real;
	return os;
}

struct Cell
{
	Molecule m[N];
};

class FMM_Grid
{
public:
	FMM_Grid()
	{
		std::default_random_engine reng(std::random_device{}());
		std::uniform_real_distribution<float> rd(0.0, 1.0);
		for (int z = 0; z < D; z++)
		{
			for (int y = 0; y < D; y++)
			{
				for (int x = 0; x < D; x++)
				{
					Cell* c = &cells[z][y][x];
					for (int p = 0; p < N; p++)
					{
						c->m[p] = { rd(reng) + x, rd(reng) + y, rd(reng) + z, 1.0, 0.0, 0.0 };
					}
				}
			}
		}
	}

	void Show()
	{
		for (int i = 0; i < D; i++)
		{
			for (int j = 0; j < D; j++)
			{
				for (int k = 0; k < D; k++)
				{
					Cell* c = &cells[i][j][k];
					for (int p = 0; p < N; p++)
					{
						std::cout << c->m[p] << std::endl;
					}
				}
			}
		}
	}

	void Real()
	{
		for (int tz = 0; tz < D; tz++)
		{
			for (int ty = 0; ty < D; ty++)
			{
				for (int tx = 0; tx < D; tx++)
				{
					for (int tp = 0; tp < N; tp++)
					{
						float u_sum = 0.0f;
						Molecule* tm = &cells[tz][ty][tx].m[tp];
						for (int sz = 0; sz < D; sz++)
						{
							for (int sy = 0; sy < D; sy++)
							{
								for (int sx = 0; sx < D; sx++)
								{
									for (int sp = 0; sp < N; sp++)
									{
										Molecule* sm = &cells[sz][sy][sx].m[sp];
										float distance = Molecule::Distance(sm, tm);
										if (distance != 0)
											u_sum += sm->q / distance;
									}
								}
							}
						}
						tm->u_real = u_sum;
					}
				}
			}
		}
	}

	void Approximate()
	{
		for (int tz = 0; tz < D; tz++)
		{
			for (int ty = 0; ty < D; ty++)
			{
				for (int tx = 0; tx < D; tx++)
				{
					for (int sz = 0; sz < D; sz++)
					{
						for (int sy = 0; sy < D; sy++)
						{
							for (int sx = 0; sx < D; sx++)
							{
								if (abs(tx - sx) < 2 && abs(ty - sy) < 2 && abs(tz - sz) < 2)  // neighbour
								{
									for (int tp = 0; tp < N; tp++)
									{
										Molecule* tm = &cells[tz][ty][tx].m[tp];
										float u_sum = 0.0f;
										for (int sp = 0; sp < N; sp++)
										{
											Molecule* sm = &cells[sz][sy][sx].m[sp];
											float distance = Molecule::Distance(sm, tm);
											if (distance != 0)
												u_sum += sm->q / distance;
										}
										tm->u_appx += u_sum;
									}
								}
								else // non-neighbour
								{
									float q_sum = 0.0f;
									for (int sp = 0; sp < N; sp++)
									{
										Molecule* sm = &cells[sz][sy][sx].m[sp];
										q_sum += sm->q;
									}
									float distance = sqrt(pow(sx - tx, 2) + pow(sy - ty, 2) + pow(sz - tz, 2));
									float u_temp = q_sum / distance;
									for (int tp = 0; tp < N; tp++)
									{
										Molecule* tm = &cells[tz][ty][tx].m[tp];
										tm->u_appx += u_temp;
									}
								}
							}
						}
					}
				}
			}
		}
	}

private:
	Cell cells[D][D][D];
};

int main()
{
	FMM_Grid grid;
	grid.Approximate();
	grid.Real();
	grid.Show();

	return 0;
}