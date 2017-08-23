#include <Rcpp.h>
#include <Rcpp/Benchmark/Timer.h>
using namespace Rcpp;

// [[Rcpp::export]]
List g_custom_objective_function_avgrank_cxx(NumericVector p_preds,
                                             IntegerVector p_labels,
                                             List p_group_to_case_ids_map_list,
                                             CharacterVector p_case_group_ids,
                                             double p_bandwidth) {
  double sqrt_pi_bw = sqrt(M_PI)*p_bandwidth;
  double bw2 = pow(p_bandwidth, 2.0);
  double tdbw2 = 2.0 / bw2;
  long ncase = p_preds.size();
  NumericVector gradients = NumericVector(ncase);
  NumericVector hessians = NumericVector(ncase);
  std::vector<std::string> case_group_ids_converted = Rcpp::as< std::vector< std::string > >(p_case_group_ids);
  NumericVector results = NumericVector::create();
  IntegerVector cases_for_comparison = IntegerVector(ncase); // pre-allocating for speed

  for (int i = 0; i < ncase; ++i) {
    // get the group case IDs
    double case_pred = p_preds[i];
    long case_label = p_labels[i];
    double case_sign = (case_label == 1) ? -1.0 : 1.0;
    std::string group_id = case_group_ids_converted[i];
    IntegerVector group_case_ids = p_group_to_case_ids_map_list[group_id];

    long neighbor_case_id = -1;
    long group_size = group_case_ids.size();
    long nneighbors = 0;

    for (long j = 0; j < group_size; ++j) {
      neighbor_case_id = group_case_ids[j];
      if (p_labels[neighbor_case_id - 1] != case_label) {
        // don't use push_back here; it is slower than direct indexing
        cases_for_comparison[nneighbors++] = neighbor_case_id;
      }
    }
    
    if (nneighbors > 0) {
      NumericVector delta_preds = NumericVector(nneighbors);
      NumericVector intermed_vals = NumericVector(nneighbors);
      NumericVector intermed2_vals = NumericVector(nneighbors);
      for (long j = 0; j < nneighbors; ++j) {
        delta_preds[j] = case_pred - p_preds[cases_for_comparison[j]-1];
        intermed_vals[j] = exp(-(delta_preds[j]*delta_preds[j] / bw2))/sqrt_pi_bw;
        intermed2_vals[j] = (fabs(delta_preds[j]) + p_bandwidth)*intermed_vals[j];
      }
      gradients[i] = case_sign * sum(intermed_vals);
      hessians[i] = tdbw2 * sum(intermed2_vals);
    }
    else {
      double probability_i = 1.0 / (1.0 + exp(-case_pred));
      gradients[i] = probability_i - (double)case_label;
      hessians[i] = probability_i * (1.0 - probability_i);
    }



  }

  return List::create(Named("grad")=gradients,
                      Named("hess")=hessians);
}
