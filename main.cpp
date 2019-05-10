#include <iostream>
#include "Eigen/Dense"
#include <math.h>
#include <vector>
#include <string>
#include <random>
#include <cstdio>
#include <ctime>
#include <list>
#include <thread>

class solver2
{
public:
	//we assume A.col(A.cols()-1) is the column corresponding to b. We want x(A.cols()-1) to be positive
	int state; //state 0:in progress, 1:
	Eigen::VectorXd x, y;
	Eigen::MatrixXd A, ATA;
	Eigen::VectorXd ATy;
	Eigen::VectorXd oricols; //the original indeces of the columns left in A
	int n, m;
	double acc;
	solver2(Eigen::MatrixXd A)
	{
		m = A.rows();
		n = A.cols();
		for (int i = 0; i < n; i++)
		{
			(this->A).col(i).normalize();
		}
		x.resize(n);
		for (int i = 0; i < n; i++)
		{
			x(i) = 1.0 / ((float)n);
		}
		y = A * x;
		ATA = (this->A).transpose() * (this->A);
		ATy = (this->A).transpose() * y;
		acc=pow(2.0,-30);
		oricols.resize(n);
		for (int i = 0; i < n; i++)
		{
			oricols(i) = i;
		}
		state=0;
	}
	void Newtonstep()
	{
		double mi = 0;
		int k = 0;
		for (int i = 0; i < n; i++)
		{
			if (ATy(i) < mi)
			{
				mi = ATy(i);
				k = i;
			}
		}
		if (mi>=0)
		{
			for(int i=0; i<n;i++)
			{
				if(ATy(i)>0)
				{
					A.block(0,i,A.rows(),A.cols()-1-i) = A.block(0,i+1,A.rows(),A.cols()-1-i);
					A.conservativeResize(A.rows(),A.cols()-1);
					oricols.segment(i,oricols.size()-1-i)=oricols.segment(i+1,oricols.size()-1-i);
					oricols.conservativeResize(oricols.size()-1);
					x.segment(i,x.size()-1-i)=x.segment(i+1,x.size()-1-i);
					x.conservativeResize(x.size()-1);
					y=A*x;
				}
			}
		}
		if (mi < 0)
		{
			double lambda=(1-mi)/(y.squaredNorm()-2*mi+1);
			x=x*lambda+(1-lambda)*Eigen::VectorXd::Unit(n, k);
			y=y*lambda+(1-lambda)*A.col(k);
			ATy = lambda*ATy - (1-lambda) * ATA.col(k);
		}
	}

};

class LPsolver
{
public:
	int n, m, state;		 //cols, rows, state: 4->running...
	int scaleT;					 //number of times scaled;
	double delta, logPa; //"error" below which we can round to find solutions, measure of flatness of the polieder
	long double ynorm;	 //"error"
	long double logynorm;
	Eigen::VectorXd x, xout, y, ATy;
	Eigen::MatrixXd A, ATA; //let the columns of A be normalized
	Eigen::MatrixXd Aori;
	std::vector<long double> Anorms; // let this hold the real norms of A
	std::vector<long double> logAnorms;
	bool big, finished;

	std::list<Eigen::VectorXd> dirs;

	LPsolver(Eigen::MatrixXd A)
	{
		state = 4;
		finished = false;
		this->A = A;
		Aori = A;
		m = A.rows();
		n = A.cols();
		Anorms.resize(n);
		logAnorms.resize(n);
		for (int i = 0; i < n; i++)
		{
			Anorms[i] = A.col(i).norm();
		}
		for (int i = 0; i < n; i++)
		{
			logAnorms[i] = log(A.col(i).norm());
		}
		x.resize(n);
		x.setOnes();
		logPa = findPa();
		delta = finddelta();
		y = A * x;
		ynorm = y.norm();
		logynorm = log(y.norm());
		y.normalize();
		for (int i = 0; i < n; i++)
		{
			(this->A).col(i).normalize();
		}
		ATA = (this->A).transpose() * (this->A);
		ATy.resize(n);
		//for(int i=0;i<n;i++){ATy(i)=A.col(i).dot(y);}
		ATy = (this->A).transpose() * y;
		big = false;
	}

	void recompute()
	{
		if (!big)
		{
			y.setZero();
			for (int i = 0; i < n; i++)
			{
				y += A.col(i) * Anorms[i] * x(i);
			}
			ynorm = y.norm();
			logynorm = log(ynorm);
			y.normalize();
		}
		ATA = A.transpose() * A;
		ATy = A.transpose() * y;
	}

	//scaling step
	void scale()
	{
		A = (A + y * ATy.transpose());
		ynorm *= 2;
		logynorm += log(2);
		std::vector<double> norms(n);
		for (int i = 0; i < n; i++)
		{
			norms[i] = A.col(i).norm();
		}
		for (int i = 0; i < n; i++)
		{
			Anorms[i] *= norms[i];
		}
		for (int i = 0; i < n; i++)
		{
			logAnorms[i] += log(norms[i]);
		}

		ATA = (ATA + 3 * ATy * ATy.transpose());
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < n; j++)
			{
				ATA(i, j) = ATA(i, j) / (norms[i] * norms[j]);
			}
		}
		for (int i = 0; i < n; i++)
		{
			A.col(i).normalize();
		}

		ATy = 2 * ATy;
		for (int i = 0; i < n; i++)
		{
			ATy(i) = ATy(i) / norms[i];
		}
		//pontosítás
		if (scaleT % (m * n) == 0)
		{
			recompute();
		}
		logPa += log(3.0 / 2.0);

		dirs.push_front(y);
		//suggestion: keep Ty and check if A.transpose()*Ty > -e where e is tiny
	}

	int solve2()
	{
		bool done = false;
		solve(done);
	}

	int solve(bool &done)
	{
		int t = 0;
		int iter = 0;

		while (logynorm >= log(delta) && logPa <= 0 && !done)
		{
			iter++;
			if (!big)
			{
				for (int i = 0; i < n; i++)
				{
					if (Anorms[i] > pow(10, 150))
					{
						big = true;
					}
				}
			}
			double mi = 0;
			int k = 0;
			for (int i = 0; i < n; i++)
			{
				if (ATy(i) < mi)
				{
					mi = ATy(i);
					k = i;
				}
			}
			if (mi == 0)
			{
				Eigen::VectorXd Ty = y;
				for (auto v : dirs)
				{
					Ty = (Eigen::MatrixXd::Identity(m, m) + v * v.transpose()) * Ty;
					Ty.normalize();
				}
				double mi2 = 100;
				std::cout << "!" << m << std::endl;
				std::cout << (Aori.transpose() * Ty).size() << std::endl;
				for (int i = 0; i < m; i++)
				{
					if ((Aori.transpose() * Ty)(i) < mi2)
					{
						mi2 = (Aori.transpose() * Ty)(i);
					}
				}
				//std::cout << "dualis megoldas pontossaga: " << mi2 << std::endl;
				//std::cout << "status: -1 " << "iter: " << iter << " rescale: " << t << " big: " << big << std::endl;
				state = 2;
				finished = true;
				done = true;
				return 1;
			}

			if (mi < -1.0 / (11 * m))
			{
				if (!big)
				{
					x = x - (mi * (ynorm / Anorms[k])) * Eigen::VectorXd::Unit(n, k);
				}
				else
				{
					x = x - (mi * exp(logynorm - logAnorms[k])) * Eigen::VectorXd::Unit(n, k);
				}

				Eigen::VectorXd y_ = y - mi * A.col(k);
				double len = y_.norm();
				ynorm *= len;
				logynorm += log(len);
				y = y_.normalized();
				ATy = (ATy - mi * ATA.col(k)) / len;
			}
			else
			{
				t++;
				scale();
			}
		}
		if (logynorm < log(delta))
		{
			for (int i = 0; i < n; i++)
			{
				if (!big)
				{
					A.col(i) *= Anorms[i];
				}
				else
				{
					A.col(i) *= exp(logAnorms[i]);
				}
			}
			y *= ynorm;
			x = x - A.transpose() * (A * A.transpose()).inverse() * y;
			xout = x;
			xout.conservativeResize(xout.size() - 1);
			for (int i = 0; i < xout.size(); i++)
			{
				xout(i) = xout(i) / x(x.size() - 1);
			}
			//std::cout << "megoldas pontossaga: " << (Aori*x).norm() << std::endl;
			for (int i = 0; i < n; i++)
			{
				if (x(i) < 0)
				{
					std::cout << "negativ koordinata!!";
				}
			}
			finished = true;
			state = 1;
			done = true;
			return 1;
		}
		else
		{
			if (done)
			{
				state = 4;
				return 4;
			}
			//std::cout << y.transpose()*A <<std::endl;
			Eigen::VectorXd Ty = y;
			for (auto v : dirs)
			{
				Ty = (Eigen::MatrixXd::Identity(m, m) + v * v.transpose()) * Ty;
				Ty.normalize();
			}
			double mi2 = 100;
			for (int i = 0; i < m; i++)
			{
				if (((Aori.transpose() * Ty).transpose())(i) < mi2)
				{
					mi2 = ((Aori.transpose() * Ty).transpose())(i);
				}
			}
			//std::cout << "dualis megoldas pontossaga: " << mi2 << std::endl;

			state = 3;
			finished = true;
			done = true;
			return 3;
		}
		//std::cout << "status: " << status << " iter: " << iter << " rescale: " << t << " big: " << big << std::endl;
	}

	double findPa()
	{
		std::vector<double> l(n);
		for (int i = 0; i < n; i++)
		{
			l[i] = A.col(i).norm();
		}
		std::sort(l.begin(), l.end(), std::greater<int>());
		double logPa = log(pow(m, 1.5));
		for (int i = 0; i < m; i++)
		{
			logPa += l[i];
		}
		logPa += l[0];
		return -m * logPa;
	}

	double finddelta()
	{
		Eigen::MatrixXd invAAT = (A * A.transpose()).inverse();
		double mi = 0;
		for (int i = 0; i < n; i++)
		{
			if ((invAAT * A.col(i)).norm() > mi)
			{
				mi = (invAAT * A.col(i)).norm();
			}
		}
		return 1 / mi;
	}
};

//make it so that the problem ha a strictly positive solution
void makeSolvable(Eigen::MatrixXd &A)
{
	Eigen::VectorXd x(A.cols());
	for (int i = 0; i < A.cols(); i++)
	{
		x(i) = 1.0 / (i + 1);
	}
	Eigen::VectorXd y = A * x;
	A.conservativeResize(A.rows(), A.cols() + 1);
	A.col(A.cols() - 1) = -y;
}

// the dual of Ax=b x>=0 is yA>=0 yb<=-1
void dual(const Eigen::MatrixXd &A, const Eigen::VectorXd &b, Eigen::MatrixXd &dA, Eigen::VectorXd &db)
{
	dA.resize(A.cols() + 1, A.rows() * 2 + A.cols() + 1);
	Eigen::MatrixXd I = Eigen::MatrixXd::Identity(A.cols(), A.cols());
	Eigen::VectorXd zeros, zeros2;
	zeros.setConstant(A.cols(), 0.0);
	zeros2.setConstant(A.cols(), 0.0);
	dA << A.transpose(), -A.transpose(), -I * 1000, zeros,
			b.transpose(), -b.transpose(), zeros2.transpose(), 1 * 1000;

	db.setConstant(A.cols() + 1, 0);
	db(A.cols()) = -1;
}

//eps for which blow up doesnt create new solutions if there werent any
double findEps(const Eigen::MatrixXd &A)
{
	std::vector<double> l(A.cols());
	for (int i = 0; i < A.cols(); i++)
	{
		l[i] = A.col(i).norm();
	}
	std::sort(l.begin(), l.end(), std::greater<int>());
	double detB = 1;
	for (int i = 0; i < A.rows(); i++)
	{
		detB *= l[i];
	}
	Eigen::VectorXd v, e;
	e.setConstant(A.cols(), 1.0);
	v = A * e;
	return 1.0 / (detB * v.lpNorm<1>());
}

//blow up polieder but doesnt create new solutions if there werent any
void blowup(const Eigen::MatrixXd &A, Eigen::VectorXd &b)
{
	double eps = findEps(A);
	Eigen::VectorXd v, e;
	e.setConstant(A.cols(), 1.0);
	v = A * e;
	b = b + v * eps;
}

//transfers b into A
void moveb(Eigen::MatrixXd &A, const Eigen::VectorXd &b)
{
	A.conservativeResize(A.rows(), A.cols() + 1);
	A.col(A.cols() - 1) = -b;
}

bool feas(Eigen::MatrixXd A, Eigen::VectorXd b, Eigen::VectorXd &x)
{
	moveb(A, b);
	LPsolver L(A);
	L.solve2();
	if (L.state == 1)
	{
		x = L.xout;
	}
}

//decides if Ax=b x>=0 is solvable solves the dual and the primal at the same time
//x is only returned if primal solvable
bool solvable(const Eigen::MatrixXd &A, const Eigen::VectorXd &b, Eigen::VectorXd &x)
{
	Eigen::MatrixXd dA;
	Eigen::VectorXd db;

	dual(A, b, dA, db);

	Eigen::VectorXd b_ = b;
	Eigen::MatrixXd A_ = A;
	blowup(A, b_);
	blowup(dA, db);
	moveb(A_, b_);
	moveb(dA, db);
	LPsolver L(A_);
	LPsolver L2(dA);
	bool done = false;
	//L.solve(done);
	std::thread first(&LPsolver::solve, &L, std::ref(done));
	L2.solve(done);
	first.join();
	if ((L.finished && L.state == 1) || (L2.finished && L2.state != 1))
	{
		x = L.xout;
		return true;
	}
	else
	{
		return false;
	}
}

//A*x=b x>=0 c*x>=val
void optiMatrix(Eigen::MatrixXd &A, Eigen::VectorXd &b, Eigen::VectorXd c, double val)
{
	Eigen::MatrixXd newA;
	Eigen::VectorXd zeros;
	zeros.setConstant(A.rows(), 0.0);
	newA.resize(A.rows() + 1, A.cols() + 1);
	newA << A, zeros,
			c.transpose(), -1;
	A = newA;
	b.conservativeResize(b.size() + 1);
	b(b.size() - 1) = val;
}

//solves A*x=b x>=0 max cx
double optimize(const Eigen::MatrixXd &A, const Eigen::VectorXd &b, const Eigen::VectorXd &c)
{
	Eigen::VectorXd x;
	if (!solvable(A, b, x))
	{
		std::cout << "not solvable" << std::endl;
		return -100000;
	}
	double val1 = c.transpose() * x;
	double val2 = 1000;
	Eigen::MatrixXd A0 = A;
	Eigen::VectorXd b0 = b;

	optiMatrix(A0, b0, c, val2);
	bool bsolv = solvable(A0, b0, x);
	while (bsolv)
	{
		val2 = val2 * 10;
		b0(b0.size() - 1) = val2;
		bsolv = solvable(A0, b0, x);
		if (val2 > 10000000)
		{
			std::cout << "likely infinite" << std::endl;
			return 10000000;
		}
	}
	while (val2 - val1 > 0.01)
	{
		b0(b0.size() - 1) = (val1 + val2) / 2.0;
		if (solvable(A0, b0, x))
		{
			val1 = (val1 + val2) / 2.0;
		}
		else
		{
			val2 = (val1 + val2) / 2.0;
		}
	}
	x.conservativeResize(x.size() - 1);
	//std::cout << "||Ax-b||: " << (A*x-b).norm() << std::endl;
	//std::cout << c.transpose()*x << std::endl;
	return val1;
}

//
void runner(std::string filename)
{
	FILE *fp2;
	fp2 = fopen(filename.c_str(), "w");
	FILE *fp;
	fp = fopen("out.txt", "r");
	int num;
	fscanf(fp, "%d", &num);
	for (int k = 1; k <= num; k++)
	{
		int n, m;
		fscanf(fp, "%d %d", &m, &n);
		fprintf(fp2, "%d\n", k);
		fprintf(fp2, "%d %d\n", m, n);
		std::cout << k << std::endl;
		Eigen::MatrixXd A(m, n - 1);
		Eigen::VectorXd b(m);
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n - 1; j++)
			{
				int in;
				fscanf(fp, "%d", &in);
				A(i, j) = in;
			}
			int in;
			fscanf(fp, "%d", &in);
			b(i) = in;
		}
		if (true)
		{
			Eigen::VectorXd c;
			c.setConstant(A.cols(), -1.0);
			std::clock_t start;
			start = std::clock();
			std::cout << "opt: " << optimize(A, b, c) << std::endl;
			fprintf(fp2, "%f\n", (std::clock() - start) / (double)CLOCKS_PER_SEC);
		}
		if (false)
		{
			Eigen::MatrixXd dA;
			Eigen::VectorXd db;

			dual(A, b, dA, db);
			//blowup(A,b);
			//blowup(dA,db);
			moveb(A, b);
			moveb(dA, db);
			LPsolver L(A);
			LPsolver L2(dA);
			std::clock_t start;
			start = std::clock();
			bool done = false;
			std::thread first(&LPsolver::solve, &L, std::ref(done));
			L2.solve(done);
			first.join();

			std::cout << (std::clock() - start) / (double)CLOCKS_PER_SEC << std::endl
								<< std::endl;
		}
		//fprintf(fp2,"%d ",stat);
		//fprintf(fp2,"%f\n",( std::clock() - start ) / (double) CLOCKS_PER_SEC);
	}
	fclose(fp);
	fclose(fp2);
}

void writer(int m, int n, int mi, int ma, int num)
{
	FILE *fp;
	fp = fopen("out.txt", "w");
	fprintf(fp, "%d\n", num);
	for (int k = 1; k <= num; k++)
	{
		std::default_random_engine generator(k);
		std::uniform_int_distribution<int> distribution(mi, ma);
		distribution(generator);
		fprintf(fp, "%d %d\n", m, n);
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n; j++)
			{
				fprintf(fp, "%d ", distribution(generator));
			}
			fprintf(fp, "\n");
		}
	}
	fclose(fp);
}

void run(std::string filename)
{
	FILE *fp;
	fp = fopen(filename.c_str(), "r");

	int n, m;
	fscanf(fp, "%d %d", &m, &n);
	Eigen::MatrixXd A(m, n - 1);
	Eigen::VectorXd b(m);
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n - 1; j++)
		{
			int in;
			fscanf(fp, "%d", &in);
			A(i, j) = in;
		}
		int in;
		fscanf(fp, "%d", &in);
		b(i) = in;
	}
	Eigen::VectorXd c;
	c.setConstant(A.cols(), -1.0);
	std::clock_t start;
	start = std::clock();
	std::cout << "opt: " << optimize(A, b, c) << std::endl;
	std::cout << "time: " << (std::clock() - start) / (double)CLOCKS_PER_SEC << std::endl;

	fclose(fp);
}

void write(std::string filename)
{
	FILE *fp;
	int mi = -100;
	int ma = 100;
	int m = 10;
	int n = 20;
	fp = fopen(filename.c_str(), "w");
	std::default_random_engine generator(0);
	std::uniform_int_distribution<int> distribution(mi, ma);
	distribution(generator);
	fprintf(fp, "%d %d\n", m + 1, n);
	for (int i = 0; i < n - 2; i++)
	{
		fprintf(fp, "%d ", 0);
	}
	fprintf(fp, "%d ", 1);
	fprintf(fp, "%d ", 0);
	fprintf(fp, "\n");
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			fprintf(fp, "%d ", distribution(generator));
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
}

int main()
{
	/* write("out.txt");
	run("out.txt"); */

	Eigen::MatrixXd A(3, 4);
	A << 0, 0, 0, 1,
			1, -1, -1, 0,
			0, -1, 1, 0;
	Eigen::VectorXd b(3);
	b << 0,
			0,
			0;
	Eigen::VectorXd c;
	c.setConstant(A.cols(), -1.0);
	std::clock_t start;
	start = std::clock();
	std::cout << "opt: " << optimize(A, b, c) << std::endl;
	std::cout << "time: " << (std::clock() - start) / (double)CLOCKS_PER_SEC << std::endl;
	/* int n;
	n = 6;
	writer(n, n * 2, -100, 100, 5);
	runner("times6.txt");
	n = 8;
	writer(n, n * 2, -100, 100, 5);
	runner("times8.txt");
	n = 10;
	writer(n, n * 2, -100, 100, 5);
	runner("times10.txt");
	n = 12;
	writer(n, n * 2, -100, 100, 5);
	runner("times12.txt");
	n = 14;
	writer(n, n * 2, -100, 100, 5);
	runner("times14.txt");
	n = 16;
	writer(n, n * 2, -100, 100, 5);
	runner("times16.txt"); */
}
