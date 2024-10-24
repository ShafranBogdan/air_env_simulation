#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <unistd.h>

class Point
{
public:
    Point(double x1, double y1): x{x1}, y{y1}, is_dekart{false} {}
    Point(double x1, double y1, int t1): x{x1}, y{y1}, t{t1}, is_dekart{false} {}
    Point(double x1, double y1, double z1): x{x1}, y{y1}, z{z1}, is_dekart{true} {}
    void to_dekart()
    {
        if(is_dekart)
            return;
        double x1 = x, y1 = y;
        x = R * sin(x1 * pi / 180) * cos(y1 * pi / 180); //переводим координаты из градусов в радианы
        y = R * sin(x1 * pi / 180) * sin(y1 * pi / 180);
        z = R * cos(x1 * pi / 180);
    }
    double get_x() {return x;}
    double get_y() {return y;}
    double get_z() {return z;}
    int get_time() {return t;}
private:
    double x{};
    double y{};
    double z{};
    int t = 0;
    double pi = 3.141592;
    int R{6400 * 1000}; //радиус Земли в м
    bool is_dekart;
};

class Grad
{
public:
    Grad(Point x1, Point y1)
    {
        int delt_time = y1.get_time() - x1.get_time();
        x = (x1.get_x() - y1.get_x()) / delt_time;
        y = (x1.get_y() - y1.get_y()) / delt_time;
        z = (x1.get_z() - y1.get_z()) / delt_time;
        time = (y1.get_time() + x1.get_time()) / 2;
    }
    double get_x() {return x;}
    double get_y() {return y;}
    double get_z() {return z;}
    int get_time() {return time;}
private:
    double x, y, z;
    int time;
};

class Loplas
{
public:
    Loplas() : x{}, y{}, z{} {};
    Loplas(Grad v1, Grad v2)
    {
        int delt_time = v2.get_time() - v1.get_time();
        x = (v2.get_x() - v1.get_x()) / delt_time;
        y = (v2.get_y() - v1.get_y()) / delt_time;
        z = (v2.get_z() - v1.get_z()) / delt_time;
    }
    double get_x() {return x;}
    double get_y() {return y;}
    double get_z() {return z;}
private:
    double x, y, z;
};

int main()
{
    int reliab{0}; //изначальная достоверность равна нулю

    std::ifstream in("C:/Users/mi/Documents/НИР/data_plane.txt");
    std::vector<Point> points;
    if (in.is_open())
    {
        double x, y;
        int t;
        while(in >> x >> y >> t)
        {
            Point p = Point(x, y, t);
            p.to_dekart();
            points.push_back(p);
        }
    }
    double norm_grad{}, norm_lopl{};
    std::vector<Grad> vels;
    Loplas lopl;
    for(int i = 0; i < points.size() - 1; ++i)
    {
        Grad grad = {points[i], points[i+1]};
        vels.push_back(grad);
        norm_grad = sqrt(grad.get_y() * grad.get_y() + grad.get_x() * grad.get_x() + grad.get_z() * grad.get_z());
        if(i > 0 && i < points.size() - 1)
        {
            lopl = Loplas(vels[i-1], vels[i]);
            norm_lopl = sqrt(lopl.get_y() * lopl.get_y() + lopl.get_x() * lopl.get_x() + lopl.get_z() * lopl.get_z());
        }
        if(norm_grad < 300)
        {
            if(reliab < 95)
                reliab += 5;
        }
        else
        {
            if(reliab >= 300)
                reliab -= 5;
        }
        if(norm_lopl >= 2)
        {
            if(reliab >= 10)
                reliab -= 10;
        }
        else
        {
            if(reliab < 95)
                reliab += 5;
        }
        std::cout << "grad = (" << grad.get_x() << " " << grad.get_y() << " " << grad.get_z() << ")" << std::endl;
        std::cout << "Norm(grad) = " << norm_grad << std::endl;
        std::cout << "loplas = (" << lopl.get_x() << " " << lopl.get_y() << " " << lopl.get_z() << ")" << std::endl;
        std::cout << "Norm(loplas) = " << norm_lopl << std::endl;
        std::cout << "Reliability = " << reliab << std::endl << std::endl;
        sleep(2);
    }
    std::cout << "The end!";
    in.close();
    return 0;
}
