#include <math.h>
#include <iostream>
#include <exception>
#include <vector>
#include <utility>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
using namespace std;
namespace py = pybind11;
double vector_distance(pair<double, double>p1, pair<double, double> p2){
  return pow((p1.first - p2.first), 2) + pow((p1.second - p2.second),2);
}
pair<double, double> vector_subtraction(pair<double, double>p1, pair<double, double> p2){
  return make_pair(p1.first-p2.first, p1.second-p2.second);
}
double vector_multiplilication(pair<double, double>p1, pair<double, double> p2){
  return p1.first*p2.first + p1.second*p2.second;
}
int vector_printer(vector<double> & vector1){
  int vec_size=vector1.size();
  cout<<"starting printing"<<"\n";
  for (int i=0; i<vec_size; i++){
    cout<<vector1[i]<<" "<<i<<"\n";
  }
  cout<<"vector printed"<<"\n";
  return 0;
}
vector<vector<double>> rdp_reduction_1D(vector<double> & times,vector<double> & timeseries, vector<vector<double>> & ans_vec, double desired_distance){
  int length=timeseries.size();
  if (length <3){
    if (length>0){
      vector<double> begin_col{times[0], timeseries[0]};
      ans_vec.push_back(begin_col);
      if (length>1){
        vector<double> end_col{times[times.size()], timeseries[times.size()]};
        ans_vec.push_back(end_col);
      }
    }
    return ans_vec;
  }
  pair<double, double> begin(times[0], timeseries[1]);
  pair<double, double> end(times[length-1], timeseries[length-1]);
  vector<double> distance(length-2, 0);
  pair<double, double> end_sub_begin;
  end_sub_begin=vector_subtraction(end, begin);
  double end_dist_begin=vector_distance(end, begin);
  pair<double, double> current_pair;
  int j=1;
  for (int i=0; i<(length-2); i++){
    current_pair.first=times[j];
    current_pair.second=timeseries[j];
    distance[i]=vector_distance(begin, current_pair)-(pow(vector_multiplilication(end_sub_begin, vector_subtraction(current_pair, begin)),2)/end_dist_begin);
    j+=1;
  }
  double max_dist=*std::max_element(distance.begin(), distance.end());
  if (max_dist<pow(desired_distance, 2)){

    vector<double> begin_col{times[0], timeseries[0]};
    vector<double> end_col{times[length-1], timeseries[length-1]};
    ans_vec.push_back(begin_col);
    ans_vec.push_back(end_col);
    return ans_vec;
  }
  int max_index=(max_element(distance.begin(), distance.end())-distance.begin())+1;
  vector<double> first_times(&times[0], &times[max_index]);
  vector<double> first_points(&timeseries[0], &timeseries[max_index]);
  vector<double> second_times(&times[max_index], &times[length-1]);
  vector<double> second_points(&timeseries[max_index], &timeseries[length-1]);
  rdp_reduction_1D(first_times, first_points, ans_vec, desired_distance);
  rdp_reduction_1D(second_times, second_points, ans_vec, desired_distance);
  return ans_vec;
}
py::object rdp_controller(std::vector<double> times, std::vector<double> timeseries, double desired_distance){
  vector<vector<double>> simplified_line;
  rdp_reduction_1D(times, timeseries, simplified_line, desired_distance);
  return py::cast(simplified_line);

}
int main(){
  return 0;
}
PYBIND11_MODULE(rdp_lines, m) {
	m.def("rdp_controller", &rdp_controller, "Reduce the number of points in a one dimensional line");
}
