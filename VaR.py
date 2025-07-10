import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, chi2
import warnings
warnings.filterwarnings('ignore')

class VaRModel:
    """
    Value at Risk Model with Historical and Parametric methods
    """
    
    def __init__(self, returns_data, confidence_level=0.95):
        """
        Initialize VaR Model
        
        Parameters:
        returns_data: pandas Series or array of returns
        confidence_level: confidence level for VaR calculation (default 0.95)
        """
        # Convert to pandas Series and handle data validation
        if isinstance(returns_data, (list, np.ndarray)):
            self.returns = pd.Series(returns_data)
        elif isinstance(returns_data, pd.Series):
            self.returns = returns_data.copy()
        else:
            raise ValueError("returns_data must be a pandas Series, list, or numpy array")
        
        # Remove any infinite or NaN values
        self.returns = self.returns.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(self.returns) == 0:
            raise ValueError("No valid returns data after cleaning")
        
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
        print(f"Initialized VaR Model with {len(self.returns)} observations")
        print(f"Confidence Level: {confidence_level*100}%")
        print(f"Alpha (significance level): {self.alpha}")
        
    def historical_var(self, window=252):
        """
        Calculate Historical VaR using empirical quantiles
        
        Parameters:
        window: rolling window size for calculation
        
        Returns:
        pandas Series of VaR estimates
        """
        print(f"Calculating Historical VaR with window size: {window}")
        
        def calc_var(x):
            # Need sufficient observations for reliable estimate
            if len(x) < max(10, int(window * 0.1)):
                return np.nan
            # Use percentile for empirical quantile
            return np.percentile(x, self.alpha * 100)
        
        var_estimates = self.returns.rolling(window=window, min_periods=int(window*0.5)).apply(
            calc_var, raw=True
        )
        
        valid_estimates = var_estimates.dropna()
        print(f"Generated {len(valid_estimates)} valid Historical VaR estimates")
        
        return var_estimates
    
    def parametric_var(self, window=252):
        """
        Calculate Parametric VaR using variance-covariance method
        
        Parameters:
        window: rolling window size for calculation
        
        Returns:
        pandas Series of VaR estimates
        """
        print(f"Calculating Parametric VaR with window size: {window}")
        
        def calc_param_var(x):
            if len(x) < max(10, int(window * 0.1)):
                return np.nan
            
            # Calculate sample statistics
            mu = np.mean(x)
            sigma = np.std(x, ddof=1)  # Use sample standard deviation
            
            # Handle zero volatility case
            if sigma == 0:
                return mu
                
            # Calculate VaR using normal distribution assumption
            var_estimate = mu + norm.ppf(self.alpha) * sigma
            return var_estimate
        
        var_estimates = self.returns.rolling(window=window, min_periods=int(window*0.5)).apply(
            calc_param_var, raw=True
        )
        
        valid_estimates = var_estimates.dropna()
        print(f"Generated {len(valid_estimates)} valid Parametric VaR estimates")
        
        return var_estimates
    
    def ewma_var(self, lambda_decay=0.94, window=252):
        """
        Calculate EWMA (Exponentially Weighted Moving Average) VaR
        
        Parameters:
        lambda_decay: decay factor for EWMA (default 0.94 for daily data)
        window: initial window size
        
        Returns:
        pandas Series of VaR estimates
        """
        print(f"Calculating EWMA VaR with lambda={lambda_decay}, window={window}")
        
        var_estimates = []
        returns_array = self.returns.values
        
        for i in range(len(returns_array)):
            if i < window:
                var_estimates.append(np.nan)
                continue
            
            try:
                # Get returns subset
                returns_subset = returns_array[max(0, i-window+1):i+1]
                n = len(returns_subset)
                
                # Calculate EWMA weights (most recent gets highest weight)
                weights = np.array([lambda_decay**(n-1-j) for j in range(n)])
                weights = weights / np.sum(weights)  # Normalize weights
                
                # Calculate EWMA mean and variance
                mu_ewma = np.sum(weights * returns_subset)
                var_ewma = np.sum(weights * (returns_subset - mu_ewma)**2)
                sigma_ewma = np.sqrt(max(var_ewma, 1e-8))  # Avoid zero volatility
                
                # Calculate VaR
                var_est = mu_ewma + norm.ppf(self.alpha) * sigma_ewma
                var_estimates.append(var_est)
                
            except Exception as e:
                print(f"Error in EWMA calculation at index {i}: {e}")
                var_estimates.append(np.nan)
        
        var_series = pd.Series(var_estimates, index=self.returns.index)
        valid_estimates = var_series.dropna()
        print(f"Generated {len(valid_estimates)} valid EWMA VaR estimates")
        
        return var_series

class VaRBacktest:
    """
    VaR Backtesting Framework with overlapping and non-overlapping methods
    """
    
    def __init__(self, returns, var_estimates, confidence_level=0.95):
        """
        Initialize backtesting framework
        
        Parameters:
        returns: actual returns
        var_estimates: VaR estimates
        confidence_level: confidence level used for VaR
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
        # Convert inputs to pandas Series and align indices
        if not isinstance(returns, pd.Series):
            returns = pd.Series(returns)
        if not isinstance(var_estimates, pd.Series):
            var_estimates = pd.Series(var_estimates, index=returns.index)
        
        # Align the series and remove NaN values
        aligned_data = pd.DataFrame({
            'returns': returns,
            'var_estimates': var_estimates
        }).dropna()
        
        if len(aligned_data) == 0:
            raise ValueError("No valid overlapping data between returns and VaR estimates")
        
        self.returns = aligned_data['returns']
        self.var_estimates = aligned_data['var_estimates']
        
        print(f"Initialized backtest with {len(self.returns)} valid observations")
        
    def calculate_violations(self):
        """
        Calculate VaR violations (exceedances)
        
        Returns:
        pandas Series of boolean violations
        """
        # VaR violation occurs when actual return is less than VaR estimate
        violations = self.returns < self.var_estimates
        return violations
    
    def overlapping_backtest(self):
        """
        Overlapping backtesting method
        
        Returns:
        dict with backtesting results
        """
        violations = self.calculate_violations()
        
        results = {
            'total_observations': len(violations),
            'total_violations': int(violations.sum()),
            'violation_rate': float(violations.mean()),
            'expected_violations': len(violations) * self.alpha,
            'expected_violation_rate': self.alpha,
            'violations_series': violations
        }
        
        print(f"Overlapping Backtest Results:")
        print(f"  Total observations: {results['total_observations']}")
        print(f"  Total violations: {results['total_violations']}")
        print(f"  Violation rate: {results['violation_rate']:.4f}")
        print(f"  Expected rate: {results['expected_violation_rate']:.4f}")
        
        return results
    
    def non_overlapping_backtest(self, window=252):
        """
        Non-overlapping backtesting method
        
        Parameters:
        window: window size for non-overlapping periods
        
        Returns:
        dict with backtesting results for each period
        """
        violations = self.calculate_violations()
        n_periods = len(violations) // window
        
        if n_periods == 0:
            print("Warning: Insufficient data for non-overlapping backtest")
            return {
                'period_results': [],
                'summary': {
                    'total_periods': 0,
                    'avg_violation_rate': np.nan,
                    'std_violation_rate': np.nan
                }
            }
        
        period_results = []
        
        for i in range(n_periods):
            start_idx = i * window
            end_idx = min((i + 1) * window, len(violations))
            period_violations = violations.iloc[start_idx:end_idx]
            
            period_result = {
                'period': i + 1,
                'start_date': str(violations.index[start_idx]),
                'end_date': str(violations.index[end_idx - 1]),
                'observations': len(period_violations),
                'violations': int(period_violations.sum()),
                'violation_rate': float(period_violations.mean()),
                'expected_violations': len(period_violations) * self.alpha,
                'expected_violation_rate': self.alpha
            }
            period_results.append(period_result)
        
        violation_rates = [p['violation_rate'] for p in period_results]
        
        summary = {
            'total_periods': n_periods,
            'avg_violation_rate': float(np.mean(violation_rates)),
            'std_violation_rate': float(np.std(violation_rates, ddof=1)) if n_periods > 1 else 0.0
        }
        
        print(f"Non-overlapping Backtest Results:")
        print(f"  Total periods: {summary['total_periods']}")
        print(f"  Average violation rate: {summary['avg_violation_rate']:.4f}")
        print(f"  Standard deviation: {summary['std_violation_rate']:.4f}")
        
        return {
            'period_results': period_results,
            'summary': summary
        }

class StatisticalExceedanceTests:
    """
    Statistical tests for VaR model validation
    """
    
    def __init__(self, violations, expected_violation_rate):
        """
        Initialize statistical tests
        
        Parameters:
        violations: boolean series of violations
        expected_violation_rate: expected violation rate (alpha)
        """
        self.violations = violations
        self.expected_rate = expected_violation_rate
        self.n_obs = len(violations)
        self.n_violations = int(violations.sum())
        self.actual_rate = self.n_violations / self.n_obs if self.n_obs > 0 else 0
        
        print(f"Statistical Tests initialized:")
        print(f"  Observations: {self.n_obs}")
        print(f"  Violations: {self.n_violations}")
        print(f"  Actual rate: {self.actual_rate:.4f}")
        print(f"  Expected rate: {self.expected_rate:.4f}")
        
    def kupiec_test(self):
        """
        Kupiec Proportion of Failures (POF) test
        Tests if the observed violation rate equals the expected rate
        
        Returns:
        dict with test results
        """
        try:
            # Handle edge cases
            if self.n_obs == 0:
                return self._create_invalid_test_result('Kupiec POF Test', 'No observations')
            
            if self.n_violations == 0:
                # No violations case
                lr_stat = -2 * self.n_obs * np.log(1 - self.expected_rate)
            elif self.n_violations == self.n_obs:
                # All violations case
                lr_stat = -2 * self.n_obs * np.log(self.expected_rate)
            else:
                # General case - likelihood ratio test
                log_likelihood_null = (
                    self.n_obs * np.log(1 - self.expected_rate) + 
                    self.n_violations * np.log(self.expected_rate / (1 - self.expected_rate))
                )
                
                log_likelihood_alt = (
                    (self.n_obs - self.n_violations) * np.log(1 - self.actual_rate) + 
                    self.n_violations * np.log(self.actual_rate)
                )
                
                lr_stat = -2 * (log_likelihood_null - log_likelihood_alt)
            
            # Ensure test statistic is non-negative
            lr_stat = max(0, lr_stat)
            
            # Chi-square test with 1 degree of freedom
            p_value = 1 - chi2.cdf(lr_stat, df=1)
            critical_value = chi2.ppf(0.95, df=1)
            
            return {
                'test_name': 'Kupiec POF Test',
                'test_statistic': float(lr_stat),
                'p_value': float(p_value),
                'critical_value': float(critical_value),
                'reject_null': lr_stat > critical_value,
                'interpretation': 'Reject model (incorrect violation rate)' if lr_stat > critical_value else 'Accept model (correct violation rate)'
            }
            
        except Exception as e:
            print(f"Error in Kupiec test: {e}")
            return self._create_invalid_test_result('Kupiec POF Test', str(e))
    
    def christoffersen_test(self):
        """
        Christoffersen Independence Test
        Tests for independence of violations (clustering)
        
        Returns:
        dict with test results
        """
        try:
            if len(self.violations) < 2:
                return self._create_invalid_test_result('Christoffersen Independence Test', 
                                                     'Insufficient observations')
            
            # Create lagged violations for transition analysis
            violations_array = self.violations.values
            violations_lag = np.roll(violations_array, 1)
            
            # Remove first observation (no lag available)
            violations_current = violations_array[1:]
            violations_previous = violations_lag[1:]
            
            # Create transition matrix
            n00 = np.sum((~violations_current) & (~violations_previous))  # No viol -> No viol
            n01 = np.sum(violations_current & (~violations_previous))     # No viol -> Viol
            n10 = np.sum((~violations_current) & violations_previous)     # Viol -> No viol
            n11 = np.sum(violations_current & violations_previous)        # Viol -> Viol
            
            # Check for insufficient transitions
            if (n01 + n11) == 0 or (n00 + n10) == 0:
                return self._create_invalid_test_result('Christoffersen Independence Test',
                                                     'Insufficient violation transitions')
            
            # Calculate transition probabilities
            pi_01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
            pi_11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
            
            # Overall violation probability
            pi = (n01 + n11) / (n00 + n01 + n10 + n11) if (n00 + n01 + n10 + n11) > 0 else 0
            
            # Likelihood ratio test statistic for independence
            if pi_01 == 0 or pi_11 == 0 or pi == 0 or pi == 1:
                lr_stat = 0
            else:
                # Log-likelihood under null hypothesis (independence)
                ll_null = (
                    n00 * np.log(1 - pi) + n01 * np.log(pi) +
                    n10 * np.log(1 - pi) + n11 * np.log(pi)
                )
                
                # Log-likelihood under alternative hypothesis
                ll_alt = (
                    n00 * np.log(1 - pi_01) + n01 * np.log(pi_01) +
                    n10 * np.log(1 - pi_11) + n11 * np.log(pi_11)
                )
                
                lr_stat = -2 * (ll_null - ll_alt)
            
            lr_stat = max(0, lr_stat)
            p_value = 1 - chi2.cdf(lr_stat, df=1)
            critical_value = chi2.ppf(0.95, df=1)
            
            return {
                'test_name': 'Christoffersen Independence Test',
                'test_statistic': float(lr_stat),
                'p_value': float(p_value),
                'critical_value': float(critical_value),
                'reject_null': lr_stat > critical_value,
                'interpretation': 'Violations are clustered (dependent)' if lr_stat > critical_value else 'Violations are independent'
            }
            
        except Exception as e:
            print(f"Error in Christoffersen test: {e}")
            return self._create_invalid_test_result('Christoffersen Independence Test', str(e))
    
    def combined_test(self):
        """
        Combined Christoffersen test (Unconditional Coverage + Independence)
        
        Returns:
        dict with test results
        """
        try:
            kupiec_result = self.kupiec_test()
            independence_result = self.christoffersen_test()
            
            # Handle invalid test cases
            if np.isnan(kupiec_result['test_statistic']) or np.isnan(independence_result['test_statistic']):
                return self._create_invalid_test_result('Combined Christoffersen Test',
                                                     'Component tests failed')
            
            # Combined test statistic
            combined_stat = kupiec_result['test_statistic'] + independence_result['test_statistic']
            p_value = 1 - chi2.cdf(combined_stat, df=2)
            critical_value = chi2.ppf(0.95, df=2)
            
            return {
                'test_name': 'Combined Christoffersen Test',
                'test_statistic': float(combined_stat),
                'p_value': float(p_value),
                'critical_value': float(critical_value),
                'reject_null': combined_stat > critical_value,
                'interpretation': 'Reject model (incorrect coverage or dependence)' if combined_stat > critical_value else 'Accept model (correct coverage and independence)'
            }
            
        except Exception as e:
            print(f"Error in Combined test: {e}")
            return self._create_invalid_test_result('Combined Christoffersen Test', str(e))
    
    def _create_invalid_test_result(self, test_name, reason):
        """Helper method to create consistent invalid test results"""
        return {
            'test_name': test_name,
            'test_statistic': np.nan,
            'p_value': np.nan,
            'critical_value': np.nan,
            'reject_null': False,
            'interpretation': f'Test invalid: {reason}'
        }

class VaRAnalysis:
    """
    Complete VaR Analysis Framework
    """
    
    def __init__(self, returns_data, confidence_level=0.95):
        """
        Initialize VaR Analysis
        
        Parameters:
        returns_data: pandas Series of returns
        confidence_level: confidence level for VaR
        """
        self.returns = returns_data
        self.confidence_level = confidence_level
        
        try:
            self.var_model = VaRModel(returns_data, confidence_level)
        except Exception as e:
            print(f"Error initializing VaR model: {e}")
            raise
        
    def run_complete_analysis(self, window=252, plot_results=True):
        """
        Run complete VaR analysis with all methods and tests
        
        Parameters:
        window: window size for calculations
        plot_results: whether to plot results
        
        Returns:
        dict with all results
        """
        print("\n" + "="*50)
        print("VaR MODEL ANALYSIS")
        print("="*50)
        
        try:
            # Calculate VaR estimates
            print("\n1. CALCULATING VaR ESTIMATES")
            print("-" * 30)
            
            hist_var = self.var_model.historical_var(window)
            param_var = self.var_model.parametric_var(window)
            ewma_var = self.var_model.ewma_var(window=window)
            
            results = {
                'var_estimates': {
                    'historical': hist_var,
                    'parametric': param_var,
                    'ewma': ewma_var
                },
                'backtests': {},
                'statistical_tests': {}
            }
            
            # Perform backtesting for each method
            methods = ['historical', 'parametric', 'ewma']
            
            print("\n2. BACKTESTING ANALYSIS")
            print("-" * 30)
            
            for method in methods:
                print(f"\n--- {method.upper()} VaR BACKTESTING ---")
                var_est = results['var_estimates'][method]
                
                try:
                    backtest = VaRBacktest(self.returns, var_est, self.confidence_level)
                    
                    # Overlapping backtest
                    overlap_results = backtest.overlapping_backtest()
                    
                    # Non-overlapping backtest
                    non_overlap_results = backtest.non_overlapping_backtest(window)
                    
                    results['backtests'][method] = {
                        'overlapping': overlap_results,
                        'non_overlapping': non_overlap_results
                    }
                    
                    # Statistical tests
                    print(f"\n--- STATISTICAL TESTS FOR {method.upper()} ---")
                    violations = overlap_results['violations_series']
                    tests = StatisticalExceedanceTests(violations, self.alpha)
                    
                    kupiec = tests.kupiec_test()
                    christoffersen = tests.christoffersen_test()
                    combined = tests.combined_test()
                    
                    print(f"Kupiec Test: {kupiec['interpretation']}")
                    print(f"  Test Statistic: {kupiec['test_statistic']:.4f}")
                    print(f"  P-value: {kupiec['p_value']:.4f}")
                    
                    print(f"Independence Test: {christoffersen['interpretation']}")
                    print(f"  Test Statistic: {christoffersen['test_statistic']:.4f}")
                    print(f"  P-value: {christoffersen['p_value']:.4f}")
                    
                    print(f"Combined Test: {combined['interpretation']}")
                    print(f"  Test Statistic: {combined['test_statistic']:.4f}")
                    print(f"  P-value: {combined['p_value']:.4f}")
                    
                    results['statistical_tests'][method] = {
                        'kupiec': kupiec,
                        'christoffersen': christoffersen,
                        'combined': combined
                    }
                    
                except Exception as e:
                    print(f"Error in backtesting {method}: {e}")
                    results['backtests'][method] = None
                    results['statistical_tests'][method] = None
            
            # Plot results if requested
            if plot_results:
                try:
                    self.plot_results(results)
                except Exception as e:
                    print(f"Error creating plots: {e}")
            
            # Print summary
            self.print_summary(results)
            
            return results
            
        except Exception as e:
            print(f"Error in complete analysis: {e}")
            raise
    
    def plot_results(self, results):
        """
        Plot VaR analysis results with error handling
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: VaR estimates comparison
            ax1 = axes[0, 0]
            
            # Find valid data range
            valid_methods = []
            for method in ['historical', 'parametric', 'ewma']:
                var_est = results['var_estimates'][method]
                if var_est is not None and not var_est.dropna().empty:
                    valid_methods.append(method)
            
            if valid_methods:
                # Get common valid index
                valid_indices = None
                for method in valid_methods:
                    var_est = results['var_estimates'][method].dropna()
                    if valid_indices is None:
                        valid_indices = var_est.index
                    else:
                        valid_indices = valid_indices.intersection(var_est.index)
                
                if len(valid_indices) > 0:
                    for method in valid_methods:
                        var_est = results['var_estimates'][method]
                        valid_data = var_est.loc[valid_indices]
                        ax1.plot(valid_indices, valid_data, label=f'{method.title()} VaR', alpha=0.8)
                    
                    ax1.set_title('VaR Estimates Comparison', fontsize=12, fontweight='bold')
                    ax1.set_ylabel('VaR')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    ax1.tick_params(axis='x', rotation=45)
            
            # Plot 2: Returns vs VaR (Best performing method)
            ax2 = axes[0, 1]
            
            best_method = 'historical'  # Default to historical
            if results['backtests'].get(best_method) is not None:
                var_est = results['var_estimates'][best_method]
                valid_idx = ~(pd.isna(self.returns) | pd.isna(var_est))
                
                if valid_idx.sum() > 0:
                    valid_returns = self.returns[valid_idx]
                    valid_var = var_est[valid_idx]
                    
                    ax2.plot(valid_returns.index, valid_returns.values, 
                            label='Returns', alpha=0.6, color='blue', linewidth=0.5)
                    ax2.plot(valid_returns.index, valid_var.values, 
                            label=f'{best_method.title()} VaR', color='red', linewidth=1.5)
                    
                    # Plot violations
                    violations = results['backtests'][best_method]['overlapping']['violations_series']
                    violation_mask = violations & valid_idx
                    if violation_mask.sum() > 0:
                        violation_returns = self.returns[violation_mask]
                        ax2.scatter(violation_returns.index, violation_returns.values, 
                                  color='red', s=15, alpha=0.8, label='Violations', zorder=5)
                    
                    ax2.set_title(f'Returns vs {best_method.title()} VaR', fontsize=12, fontweight='bold')
                    ax2.set_ylabel('Return/VaR')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    ax2.tick_params(axis='x', rotation=45)
            
            # Plot 3: Violation rates comparison
            ax3 = axes[1, 0]
            
            methods_with_data = []
            violation_rates = []
            expected_rates = []
            
            for method in ['historical', 'parametric', 'ewma']:
                backtest_result = results['backtests'].get(method)
                if backtest_result is not None and backtest_result['overlapping'] is not None:
                    methods_with_data.append(method)
                    violation_rates.append(backtest_result['overlapping']['violation_rate'])
                    expected_rates.append(backtest_result['overlapping']['expected_violation_rate'])
            
            if methods_with_data:
                x = np.arange(len(methods_with_data))
                width = 0.35
                
                ax3.bar(x - width/2, violation_rates, width, label='Actual', alpha=0.8, color='skyblue')
                ax3.bar(x + width/2, expected_rates, width, label='Expected', alpha=0.8, color='orange')
                ax3.set_xlabel('VaR Method')
                ax3.set_ylabel('Violation Rate')
                ax3.set_title('Violation Rates Comparison', fontsize=12, fontweight='bold')
                ax3.set_xticks(x)
                ax3.set_xticklabels([m.title() for m in methods_with_data])
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            
            # Plot 4: Statistical test p-values
            ax4 = axes[1, 1]
            
            test_types = ['kupiec', 'christoffersen', 'combined']
            test_labels = ['Kupiec', 'Independence', 'Combined']
            
            methods_with_tests = []
            p_values_by_method = {test: [] for test in test_types}
            
            for method in ['historical', 'parametric', 'ewma']:
                test_results = results['statistical_tests'].get(method)
                if test_results is not None:
                    methods_with_tests.append(method)
                    for test_type in test_types:
                        p_val = test_results.get(test_type, {}).get('p_value', np.nan)
                        p_values_by_method[test_type].append(p_val if not np.isnan(p_val) else 0)
            
            if methods_with_tests:
                x = np.arange(len(test_labels))
                width = 0.25
                colors = ['lightcoral', 'lightgreen', 'lightblue']
                
                for i, method in enumerate(methods_with_tests):
                    p_vals = [p_values_by_method[test][i] for test in test_types]
                    ax4.bar(x + i*width - width, p_vals, width, 
                           label=method.title(), alpha=0.8, color=colors[i])
                
                ax4.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='5% Significance')
                ax4.set_xlabel('Statistical Test')
                ax4.set_ylabel('P-value')
                ax4.set_title('Statistical Test Results', fontsize=12, fontweight='bold')
                ax4.set_xticks(x)
                ax4.set_xticklabels(test_labels)
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                ax4.set_ylim(0, max(1, ax4.get_
