"""
Financial Health Agent for fundamental company analysis
"""

from typing import Dict, List, Any, Optional, Tuple
import asyncio
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from collections import defaultdict
import json

from .base_agent import BaseAgent, SharedMemory, AgentCommunication
from ..opencog_framework.truth_values import TruthValue


class FinancialHealthAgent(BaseAgent):
    """Agent specialized in fundamental financial analysis and health assessment"""
    
    def __init__(self, agent_id: str, shared_memory: SharedMemory, 
                 communication: AgentCommunication, config: Dict[str, Any] = None):
        super().__init__(agent_id, shared_memory, communication, config)
        
        self.financial_metrics = self._initialize_financial_metrics()
        self.industry_benchmarks = config.get('industry_benchmarks', {})
        self.risk_thresholds = self._initialize_risk_thresholds()
        
    def _initialize_financial_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Initialize financial metric definitions and calculation methods"""
        return {
            'profitability': {
                'metrics': ['gross_profit_margin', 'operating_margin', 'net_profit_margin', 'roe', 'roa', 'roic'],
                'weight': 0.25,
                'description': 'Company ability to generate profit from operations'
            },
            'liquidity': {
                'metrics': ['current_ratio', 'quick_ratio', 'cash_ratio', 'operating_cash_flow_ratio'],
                'weight': 0.20,
                'description': 'Company ability to meet short-term obligations'
            },
            'leverage': {
                'metrics': ['debt_to_equity', 'debt_to_assets', 'interest_coverage_ratio', 'debt_service_coverage'],
                'weight': 0.20,
                'description': 'Company debt management and financial leverage'
            },
            'efficiency': {
                'metrics': ['asset_turnover', 'inventory_turnover', 'receivables_turnover', 'working_capital_turnover'],
                'weight': 0.15,
                'description': 'How efficiently company uses its assets'
            },
            'growth': {
                'metrics': ['revenue_growth', 'earnings_growth', 'book_value_growth', 'free_cash_flow_growth'],
                'weight': 0.20,
                'description': 'Company growth trajectory and sustainability'
            }
        }
    
    def _initialize_risk_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialize risk threshold levels for different metrics"""
        return {
            'debt_to_equity': {'low': 0.3, 'medium': 0.6, 'high': 1.0},
            'current_ratio': {'low': 2.0, 'medium': 1.5, 'high': 1.0},
            'interest_coverage_ratio': {'low': 10.0, 'medium': 5.0, 'high': 2.0},
            'operating_margin': {'low': 0.15, 'medium': 0.10, 'high': 0.05},
            'roe': {'low': 0.15, 'medium': 0.10, 'high': 0.05}
        }
    
    def get_required_data(self) -> List[str]:
        """Return required data types for financial health analysis"""
        return [
            'annual_reports', 'quarterly_reports', 'financial_statements',
            'cash_flow_statements', 'balance_sheets', 'income_statements',
            'stock_ticker', 'industry'
        ]
    
    async def analyze(self, company: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive financial health analysis"""
        self.logger.info(f"Starting financial health analysis for {company}")
        
        results = {
            'financial_health_score': 0.0,
            'financial_metrics': {},
            'ratio_analysis': {},
            'trend_analysis': {},
            'risk_assessment': {},
            'swot_analysis': {},
            'peer_comparison': {},
            'future_outlook': {},
            'red_flags': [],
            'strengths': [],
            'confidence_scores': {}
        }
        
        # Extract financial data from reports
        financial_data = await self._extract_financial_data(data, company)
        if not financial_data:
            self.logger.warning(f"Insufficient financial data for {company}")
            return results
        
        # Calculate key financial metrics
        results['financial_metrics'] = await self._calculate_financial_metrics(financial_data)
        
        # Perform ratio analysis
        results['ratio_analysis'] = self._perform_ratio_analysis(results['financial_metrics'])
        
        # Analyze trends over time
        results['trend_analysis'] = self._analyze_financial_trends(financial_data)
        
        # Assess financial risks
        results['risk_assessment'] = self._assess_financial_risks(results['financial_metrics'])
        
        # Generate SWOT analysis
        results['swot_analysis'] = self._generate_swot_analysis(
            results['financial_metrics'], results['trend_analysis'], results['risk_assessment']
        )
        
        # Calculate overall financial health score
        results['financial_health_score'] = self._calculate_health_score(results)
        
        # Identify red flags and strengths
        results['red_flags'] = self._identify_red_flags(results)
        results['strengths'] = self._identify_strengths(results)
        
        # Generate future outlook
        results['future_outlook'] = self._generate_future_outlook(results)
        
        # Calculate confidence scores
        results['confidence_scores'] = self._calculate_confidence_scores(financial_data, results)
        
        self.logger.info(f"Financial health analysis completed for {company}")
        return results
    
    async def _extract_financial_data(self, data: Dict[str, Any], company: str) -> Dict[str, Any]:
        """Extract and structure financial data from various sources"""
        financial_data = {
            'income_statements': [],
            'balance_sheets': [],
            'cash_flow_statements': [],
            'key_metrics': {},
            'periods': []
        }
        
        # Process annual reports
        if 'annual_reports' in data:
            for report in data['annual_reports']:
                extracted_data = await self._extract_from_annual_report(report)
                if extracted_data:
                    self._merge_financial_data(financial_data, extracted_data)
        
        # Process quarterly reports
        if 'quarterly_reports' in data:
            for report in data['quarterly_reports']:
                extracted_data = await self._extract_from_quarterly_report(report)
                if extracted_data:
                    self._merge_financial_data(financial_data, extracted_data)
        
        # Process structured financial statements if available
        statement_types = ['financial_statements', 'income_statements', 'balance_sheets', 'cash_flow_statements']
        for stmt_type in statement_types:
            if stmt_type in data and data[stmt_type]:
                self._process_structured_statements(financial_data, data[stmt_type], stmt_type)
        
        return financial_data
    
    async def _extract_from_annual_report(self, report: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract financial data from annual report text"""
        text = report.get('text', '') or report.get('content', '')
        if not text:
            return None
        
        extracted_data = {
            'revenue': self._extract_metric(text, ['revenue', 'net sales', 'total revenue']),
            'net_income': self._extract_metric(text, ['net income', 'net earnings', 'profit']),
            'total_assets': self._extract_metric(text, ['total assets']),
            'total_equity': self._extract_metric(text, ['total equity', 'shareholders equity']),
            'total_debt': self._extract_metric(text, ['total debt', 'long-term debt']),
            'cash': self._extract_metric(text, ['cash and equivalents', 'cash']),
            'operating_cash_flow': self._extract_metric(text, ['operating cash flow', 'cash from operations']),
            'period': self._extract_period(report)
        }
        
        return extracted_data
    
    async def _extract_from_quarterly_report(self, report: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract financial data from quarterly report"""
        # Similar to annual report but with quarterly context
        return await self._extract_from_annual_report(report)
    
    def _extract_metric(self, text: str, metric_names: List[str]) -> Optional[float]:
        """Extract a specific financial metric from text using regex"""
        text_lower = text.lower()
        
        for metric_name in metric_names:
            # Pattern to find financial figures
            patterns = [
                rf"{metric_name.lower()}[:\s]+\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:million|billion)?",
                rf"{metric_name.lower()}[:\s\(\)]+([0-9,]+(?:\.[0-9]+)?)\s*(?:million|billion)?",
                rf"(?:{metric_name.lower()})\s*[:\$]\s*([0-9,]+(?:\.[0-9]+)?)"
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                if matches:
                    try:
                        # Take the first match and convert to float
                        value_str = matches[0].replace(',', '')
                        value = float(value_str)
                        
                        # Check for scale indicators
                        context = text_lower[max(0, text_lower.find(matches[0]) - 100):
                                           text_lower.find(matches[0]) + 100]
                        
                        if 'billion' in context:
                            value *= 1e9
                        elif 'million' in context:
                            value *= 1e6
                        
                        return value
                    except ValueError:
                        continue
        
        return None
    
    def _extract_period(self, report: Dict[str, Any]) -> Optional[str]:
        """Extract the reporting period from report metadata"""
        period_fields = ['period', 'year', 'date', 'fiscal_year']
        
        for field in period_fields:
            if field in report and report[field]:
                return str(report[field])
        
        # Try to extract from title or content
        text = (report.get('title', '') + ' ' + report.get('text', '')[:500]).lower()
        
        # Look for year patterns
        year_matches = re.findall(r'20[0-9]{2}', text)
        if year_matches:
            return year_matches[0]
        
        return None
    
    def _merge_financial_data(self, main_data: Dict[str, Any], new_data: Dict[str, Any]):
        """Merge new financial data into the main dataset"""
        period = new_data.get('period')
        if not period:
            return
        
        if period not in main_data['periods']:
            main_data['periods'].append(period)
        
        # Store the data indexed by period
        for key, value in new_data.items():
            if key != 'period' and value is not None:
                if key not in main_data['key_metrics']:
                    main_data['key_metrics'][key] = {}
                main_data['key_metrics'][key][period] = value
    
    def _process_structured_statements(self, financial_data: Dict[str, Any], 
                                     statements: List[Dict[str, Any]], stmt_type: str):
        """Process structured financial statement data"""
        for statement in statements:
            # Handle structured financial data format
            period = statement.get('period') or statement.get('year')
            if not period:
                continue
            
            if period not in financial_data['periods']:
                financial_data['periods'].append(period)
            
            # Extract metrics based on statement type
            if stmt_type in ['income_statements', 'financial_statements']:
                self._process_income_statement(financial_data, statement, period)
            elif stmt_type == 'balance_sheets':
                self._process_balance_sheet(financial_data, statement, period)
            elif stmt_type == 'cash_flow_statements':
                self._process_cash_flow_statement(financial_data, statement, period)
    
    def _process_income_statement(self, financial_data: Dict[str, Any], 
                                statement: Dict[str, Any], period: str):
        """Process income statement data"""
        income_metrics = ['revenue', 'gross_profit', 'operating_income', 'net_income', 'ebitda']
        
        for metric in income_metrics:
            value = statement.get(metric)
            if value is not None:
                if metric not in financial_data['key_metrics']:
                    financial_data['key_metrics'][metric] = {}
                financial_data['key_metrics'][metric][period] = float(value)
    
    def _process_balance_sheet(self, financial_data: Dict[str, Any], 
                             statement: Dict[str, Any], period: str):
        """Process balance sheet data"""
        balance_metrics = ['total_assets', 'total_liabilities', 'total_equity', 'cash', 'total_debt']
        
        for metric in balance_metrics:
            value = statement.get(metric)
            if value is not None:
                if metric not in financial_data['key_metrics']:
                    financial_data['key_metrics'][metric] = {}
                financial_data['key_metrics'][metric][period] = float(value)
    
    def _process_cash_flow_statement(self, financial_data: Dict[str, Any], 
                                   statement: Dict[str, Any], period: str):
        """Process cash flow statement data"""
        cf_metrics = ['operating_cash_flow', 'investing_cash_flow', 'financing_cash_flow', 'free_cash_flow']
        
        for metric in cf_metrics:
            value = statement.get(metric)
            if value is not None:
                if metric not in financial_data['key_metrics']:
                    financial_data['key_metrics'][metric] = {}
                financial_data['key_metrics'][metric][period] = float(value)
    
    async def _calculate_financial_metrics(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive financial metrics"""
        metrics = {
            'profitability': {},
            'liquidity': {},
            'leverage': {},
            'efficiency': {},
            'growth': {},
            'valuation': {}
        }
        
        key_metrics = financial_data.get('key_metrics', {})
        periods = sorted(financial_data.get('periods', []))
        
        if not periods:
            return metrics
        
        latest_period = periods[-1]
        
        # Calculate profitability metrics
        metrics['profitability'] = self._calculate_profitability_metrics(key_metrics, latest_period)
        
        # Calculate liquidity metrics
        metrics['liquidity'] = self._calculate_liquidity_metrics(key_metrics, latest_period)
        
        # Calculate leverage metrics
        metrics['leverage'] = self._calculate_leverage_metrics(key_metrics, latest_period)
        
        # Calculate efficiency metrics
        metrics['efficiency'] = self._calculate_efficiency_metrics(key_metrics, latest_period)
        
        # Calculate growth metrics (requires multiple periods)
        if len(periods) > 1:
            metrics['growth'] = self._calculate_growth_metrics(key_metrics, periods)
        
        return metrics
    
    def _calculate_profitability_metrics(self, metrics: Dict[str, Dict[str, float]], 
                                       period: str) -> Dict[str, float]:
        """Calculate profitability ratios"""
        profitability = {}
        
        revenue = metrics.get('revenue', {}).get(period)
        gross_profit = metrics.get('gross_profit', {}).get(period)
        operating_income = metrics.get('operating_income', {}).get(period)
        net_income = metrics.get('net_income', {}).get(period)
        total_assets = metrics.get('total_assets', {}).get(period)
        total_equity = metrics.get('total_equity', {}).get(period)
        
        # Gross Profit Margin
        if revenue and gross_profit and revenue > 0:
            profitability['gross_profit_margin'] = gross_profit / revenue
        
        # Operating Margin
        if revenue and operating_income and revenue > 0:
            profitability['operating_margin'] = operating_income / revenue
        
        # Net Profit Margin
        if revenue and net_income and revenue > 0:
            profitability['net_profit_margin'] = net_income / revenue
        
        # Return on Assets (ROA)
        if total_assets and net_income and total_assets > 0:
            profitability['roa'] = net_income / total_assets
        
        # Return on Equity (ROE)
        if total_equity and net_income and total_equity > 0:
            profitability['roe'] = net_income / total_equity
        
        return profitability
    
    def _calculate_liquidity_metrics(self, metrics: Dict[str, Dict[str, float]], 
                                   period: str) -> Dict[str, float]:
        """Calculate liquidity ratios"""
        liquidity = {}
        
        current_assets = metrics.get('current_assets', {}).get(period)
        current_liabilities = metrics.get('current_liabilities', {}).get(period)
        cash = metrics.get('cash', {}).get(period)
        inventory = metrics.get('inventory', {}).get(period)
        
        # Current Ratio
        if current_assets and current_liabilities and current_liabilities > 0:
            liquidity['current_ratio'] = current_assets / current_liabilities
        
        # Quick Ratio
        if current_assets and current_liabilities and inventory and current_liabilities > 0:
            liquidity['quick_ratio'] = (current_assets - inventory) / current_liabilities
        
        # Cash Ratio
        if cash and current_liabilities and current_liabilities > 0:
            liquidity['cash_ratio'] = cash / current_liabilities
        
        return liquidity
    
    def _calculate_leverage_metrics(self, metrics: Dict[str, Dict[str, float]], 
                                  period: str) -> Dict[str, float]:
        """Calculate leverage/debt ratios"""
        leverage = {}
        
        total_debt = metrics.get('total_debt', {}).get(period)
        total_equity = metrics.get('total_equity', {}).get(period)
        total_assets = metrics.get('total_assets', {}).get(period)
        operating_income = metrics.get('operating_income', {}).get(period)
        interest_expense = metrics.get('interest_expense', {}).get(period)
        
        # Debt-to-Equity Ratio
        if total_debt and total_equity and total_equity > 0:
            leverage['debt_to_equity'] = total_debt / total_equity
        
        # Debt-to-Assets Ratio
        if total_debt and total_assets and total_assets > 0:
            leverage['debt_to_assets'] = total_debt / total_assets
        
        # Interest Coverage Ratio
        if operating_income and interest_expense and interest_expense > 0:
            leverage['interest_coverage_ratio'] = operating_income / interest_expense
        
        return leverage
    
    def _calculate_efficiency_metrics(self, metrics: Dict[str, Dict[str, float]], 
                                    period: str) -> Dict[str, float]:
        """Calculate efficiency ratios"""
        efficiency = {}
        
        revenue = metrics.get('revenue', {}).get(period)
        total_assets = metrics.get('total_assets', {}).get(period)
        inventory = metrics.get('inventory', {}).get(period)
        cost_of_goods_sold = metrics.get('cost_of_goods_sold', {}).get(period)
        
        # Asset Turnover
        if revenue and total_assets and total_assets > 0:
            efficiency['asset_turnover'] = revenue / total_assets
        
        # Inventory Turnover
        if cost_of_goods_sold and inventory and inventory > 0:
            efficiency['inventory_turnover'] = cost_of_goods_sold / inventory
        
        return efficiency
    
    def _calculate_growth_metrics(self, metrics: Dict[str, Dict[str, float]], 
                                periods: List[str]) -> Dict[str, float]:
        """Calculate growth rates over time"""
        growth = {}
        
        if len(periods) < 2:
            return growth
        
        latest_period = periods[-1]
        previous_period = periods[-2]
        
        growth_metrics = ['revenue', 'net_income', 'total_assets', 'free_cash_flow']
        
        for metric in growth_metrics:
            if metric in metrics:
                latest_value = metrics[metric].get(latest_period)
                previous_value = metrics[metric].get(previous_period)
                
                if latest_value and previous_value and previous_value != 0:
                    growth_rate = (latest_value - previous_value) / previous_value
                    growth[f'{metric}_growth'] = growth_rate
        
        return growth
    
    def _perform_ratio_analysis(self, financial_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Perform comprehensive ratio analysis"""
        ratio_analysis = {
            'profitability_assessment': 'neutral',
            'liquidity_assessment': 'neutral',
            'leverage_assessment': 'neutral',
            'efficiency_assessment': 'neutral',
            'overall_assessment': 'neutral',
            'key_ratios': {},
            'benchmark_comparison': {}
        }
        
        # Assess each category
        for category, ratios in financial_metrics.items():
            if ratios:
                assessment = self._assess_ratio_category(category, ratios)
                ratio_analysis[f'{category}_assessment'] = assessment
                ratio_analysis['key_ratios'].update(ratios)
        
        # Overall assessment
        assessments = [
            ratio_analysis['profitability_assessment'],
            ratio_analysis['liquidity_assessment'],
            ratio_analysis['leverage_assessment'],
            ratio_analysis['efficiency_assessment']
        ]
        
        positive_count = assessments.count('strong') + assessments.count('good')
        negative_count = assessments.count('weak') + assessments.count('poor')
        
        if positive_count > negative_count:
            ratio_analysis['overall_assessment'] = 'positive'
        elif negative_count > positive_count:
            ratio_analysis['overall_assessment'] = 'negative'
        
        return ratio_analysis
    
    def _assess_ratio_category(self, category: str, ratios: Dict[str, float]) -> str:
        """Assess a category of financial ratios"""
        if category == 'profitability':
            return self._assess_profitability(ratios)
        elif category == 'liquidity':
            return self._assess_liquidity(ratios)
        elif category == 'leverage':
            return self._assess_leverage(ratios)
        elif category == 'efficiency':
            return self._assess_efficiency(ratios)
        elif category == 'growth':
            return self._assess_growth(ratios)
        
        return 'neutral'
    
    def _assess_profitability(self, ratios: Dict[str, float]) -> str:
        """Assess profitability ratios"""
        score = 0
        count = 0
        
        # ROE assessment
        roe = ratios.get('roe', 0)
        if roe > 0.15:  # 15%
            score += 2
        elif roe > 0.10:  # 10%
            score += 1
        elif roe < 0.05:  # 5%
            score -= 1
        count += 1
        
        # Net Profit Margin
        npm = ratios.get('net_profit_margin', 0)
        if npm > 0.10:  # 10%
            score += 2
        elif npm > 0.05:  # 5%
            score += 1
        elif npm < 0.02:  # 2%
            score -= 1
        count += 1
        
        if count == 0:
            return 'neutral'
        
        avg_score = score / count
        
        if avg_score >= 1.5:
            return 'strong'
        elif avg_score >= 0.5:
            return 'good'
        elif avg_score <= -0.5:
            return 'weak'
        else:
            return 'neutral'
    
    def _assess_liquidity(self, ratios: Dict[str, float]) -> str:
        """Assess liquidity ratios"""
        current_ratio = ratios.get('current_ratio', 0)
        
        if current_ratio >= 2.0:
            return 'strong'
        elif current_ratio >= 1.5:
            return 'good'
        elif current_ratio >= 1.0:
            return 'neutral'
        else:
            return 'weak'
    
    def _assess_leverage(self, ratios: Dict[str, float]) -> str:
        """Assess leverage ratios"""
        debt_to_equity = ratios.get('debt_to_equity', 0)
        
        if debt_to_equity <= 0.3:
            return 'strong'
        elif debt_to_equity <= 0.6:
            return 'good'
        elif debt_to_equity <= 1.0:
            return 'neutral'
        else:
            return 'weak'
    
    def _assess_efficiency(self, ratios: Dict[str, float]) -> str:
        """Assess efficiency ratios"""
        asset_turnover = ratios.get('asset_turnover', 0)
        
        if asset_turnover >= 1.5:
            return 'strong'
        elif asset_turnover >= 1.0:
            return 'good'
        elif asset_turnover >= 0.5:
            return 'neutral'
        else:
            return 'weak'
    
    def _assess_growth(self, ratios: Dict[str, float]) -> str:
        """Assess growth ratios"""
        revenue_growth = ratios.get('revenue_growth', 0)
        
        if revenue_growth >= 0.15:  # 15%
            return 'strong'
        elif revenue_growth >= 0.10:  # 10%
            return 'good'
        elif revenue_growth >= 0.05:  # 5%
            return 'neutral'
        else:
            return 'weak'
    
    def _analyze_financial_trends(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze financial trends over time"""
        trend_analysis = {
            'revenue_trend': 'stable',
            'profitability_trend': 'stable',
            'debt_trend': 'stable',
            'cash_trend': 'stable',
            'overall_trend': 'stable'
        }
        
        periods = sorted(financial_data.get('periods', []))
        key_metrics = financial_data.get('key_metrics', {})
        
        if len(periods) < 3:
            return trend_analysis
        
        # Analyze revenue trend
        if 'revenue' in key_metrics:
            revenue_values = [key_metrics['revenue'].get(p, 0) for p in periods[-3:]]
            trend_analysis['revenue_trend'] = self._calculate_trend(revenue_values)
        
        # Analyze profitability trend
        if 'net_income' in key_metrics:
            income_values = [key_metrics['net_income'].get(p, 0) for p in periods[-3:]]
            trend_analysis['profitability_trend'] = self._calculate_trend(income_values)
        
        # Analyze debt trend
        if 'total_debt' in key_metrics:
            debt_values = [key_metrics['total_debt'].get(p, 0) for p in periods[-3:]]
            trend_analysis['debt_trend'] = self._calculate_trend(debt_values, inverse=True)
        
        # Analyze cash trend
        if 'cash' in key_metrics:
            cash_values = [key_metrics['cash'].get(p, 0) for p in periods[-3:]]
            trend_analysis['cash_trend'] = self._calculate_trend(cash_values)
        
        return trend_analysis
    
    def _calculate_trend(self, values: List[float], inverse: bool = False) -> str:
        """Calculate trend direction from a series of values"""
        if len(values) < 2:
            return 'stable'
        
        # Calculate percentage changes
        changes = []
        for i in range(1, len(values)):
            if values[i-1] != 0:
                change = (values[i] - values[i-1]) / values[i-1]
                changes.append(change)
        
        if not changes:
            return 'stable'
        
        avg_change = np.mean(changes)
        
        # Adjust for inverse relationships (e.g., debt increase is bad)
        if inverse:
            avg_change = -avg_change
        
        if avg_change > 0.05:  # 5% average improvement
            return 'improving'
        elif avg_change < -0.05:  # 5% average decline
            return 'declining'
        else:
            return 'stable'
    
    def _assess_financial_risks(self, financial_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Assess various financial risks"""
        risk_assessment = {
            'liquidity_risk': 'low',
            'credit_risk': 'low', 
            'operational_risk': 'medium',
            'market_risk': 'medium',
            'overall_risk': 'medium',
            'risk_factors': []
        }
        
        # Assess liquidity risk
        liquidity_ratios = financial_metrics.get('liquidity', {})
        if liquidity_ratios:
            current_ratio = liquidity_ratios.get('current_ratio', 2.0)
            if current_ratio < 1.0:
                risk_assessment['liquidity_risk'] = 'high'
                risk_assessment['risk_factors'].append('Low current ratio indicates liquidity stress')
            elif current_ratio < 1.5:
                risk_assessment['liquidity_risk'] = 'medium'
        
        # Assess credit risk
        leverage_ratios = financial_metrics.get('leverage', {})
        if leverage_ratios:
            debt_to_equity = leverage_ratios.get('debt_to_equity', 0)
            interest_coverage = leverage_ratios.get('interest_coverage_ratio', 10)
            
            if debt_to_equity > 1.0 or interest_coverage < 2.0:
                risk_assessment['credit_risk'] = 'high'
                risk_assessment['risk_factors'].append('High leverage or weak interest coverage')
            elif debt_to_equity > 0.6 or interest_coverage < 5.0:
                risk_assessment['credit_risk'] = 'medium'
        
        # Calculate overall risk
        risk_levels = [
            risk_assessment['liquidity_risk'],
            risk_assessment['credit_risk'],
            risk_assessment['operational_risk'],
            risk_assessment['market_risk']
        ]
        
        high_risk_count = risk_levels.count('high')
        medium_risk_count = risk_levels.count('medium')
        
        if high_risk_count >= 2:
            risk_assessment['overall_risk'] = 'high'
        elif high_risk_count >= 1 or medium_risk_count >= 3:
            risk_assessment['overall_risk'] = 'medium'
        else:
            risk_assessment['overall_risk'] = 'low'
        
        return risk_assessment
    
    def _generate_swot_analysis(self, financial_metrics: Dict[str, Dict[str, float]],
                               trend_analysis: Dict[str, Any],
                               risk_assessment: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate SWOT analysis based on financial data"""
        swot = {
            'strengths': [],
            'weaknesses': [],
            'opportunities': [],
            'threats': []
        }
        
        # Identify strengths
        profitability = financial_metrics.get('profitability', {})
        if profitability.get('roe', 0) > 0.15:
            swot['strengths'].append('Strong return on equity indicating efficient use of shareholder funds')
        
        if profitability.get('net_profit_margin', 0) > 0.10:
            swot['strengths'].append('High net profit margin shows strong cost control')
        
        liquidity = financial_metrics.get('liquidity', {})
        if liquidity.get('current_ratio', 0) > 2.0:
            swot['strengths'].append('Strong liquidity position provides financial flexibility')
        
        # Identify weaknesses
        leverage = financial_metrics.get('leverage', {})
        if leverage.get('debt_to_equity', 0) > 1.0:
            swot['weaknesses'].append('High debt levels may constrain financial flexibility')
        
        if risk_assessment['overall_risk'] == 'high':
            swot['weaknesses'].append('High overall financial risk profile')
        
        # Identify opportunities
        growth = financial_metrics.get('growth', {})
        if growth.get('revenue_growth', 0) > 0.10:
            swot['opportunities'].append('Strong revenue growth indicates market opportunity')
        
        if trend_analysis.get('profitability_trend') == 'improving':
            swot['opportunities'].append('Improving profitability trend suggests operational efficiency gains')
        
        # Identify threats
        if trend_analysis.get('revenue_trend') == 'declining':
            swot['threats'].append('Declining revenue trend indicates market or competitive challenges')
        
        if risk_assessment['liquidity_risk'] == 'high':
            swot['threats'].append('Liquidity risk may impact ability to meet obligations')
        
        return swot
    
    def _calculate_health_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall financial health score (0-100)"""
        score = 0.0
        weights = self.financial_metrics
        
        # Score each category
        for category, config in weights.items():
            category_score = self._score_category(results['financial_metrics'].get(category, {}), category)
            score += category_score * config['weight']
        
        # Adjust for risk factors
        risk_level = results['risk_assessment'].get('overall_risk', 'medium')
        if risk_level == 'high':
            score *= 0.8
        elif risk_level == 'low':
            score *= 1.1
        
        # Adjust for trends
        trend_analysis = results.get('trend_analysis', {})
        positive_trends = sum(1 for trend in trend_analysis.values() if trend == 'improving')
        negative_trends = sum(1 for trend in trend_analysis.values() if trend == 'declining')
        
        trend_adjustment = (positive_trends - negative_trends) * 0.05
        score += trend_adjustment
        
        return min(max(score * 100, 0), 100)
    
    def _score_category(self, metrics: Dict[str, float], category: str) -> float:
        """Score a specific category of metrics"""
        if not metrics:
            return 0.5  # Neutral score for missing data
        
        assessment = self._assess_ratio_category(category, metrics)
        
        score_mapping = {
            'strong': 1.0,
            'good': 0.8,
            'neutral': 0.6,
            'weak': 0.4,
            'poor': 0.2
        }
        
        return score_mapping.get(assessment, 0.5)
    
    def _identify_red_flags(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify financial red flags"""
        red_flags = []
        
        financial_metrics = results.get('financial_metrics', {})
        risk_assessment = results.get('risk_assessment', {})
        
        # High debt levels
        leverage = financial_metrics.get('leverage', {})
        if leverage.get('debt_to_equity', 0) > 1.5:
            red_flags.append({
                'flag': 'Very High Debt Levels',
                'description': 'Debt-to-equity ratio exceeds 1.5, indicating potential financial stress',
                'severity': 'high'
            })
        
        # Poor liquidity
        liquidity = financial_metrics.get('liquidity', {})
        if liquidity.get('current_ratio', 2.0) < 1.0:
            red_flags.append({
                'flag': 'Liquidity Crisis',
                'description': 'Current ratio below 1.0 suggests difficulty meeting short-term obligations',
                'severity': 'critical'
            })
        
        # Declining profitability
        profitability = financial_metrics.get('profitability', {})
        if profitability.get('net_profit_margin', 0) < 0:
            red_flags.append({
                'flag': 'Operating Losses',
                'description': 'Negative net profit margin indicates operational challenges',
                'severity': 'high'
            })
        
        return red_flags
    
    def _identify_strengths(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify financial strengths"""
        strengths = []
        
        financial_metrics = results.get('financial_metrics', {})
        
        # Strong profitability
        profitability = financial_metrics.get('profitability', {})
        if profitability.get('roe', 0) > 0.20:
            strengths.append({
                'strength': 'Exceptional Profitability',
                'description': 'ROE above 20% demonstrates excellent capital efficiency',
                'metric': f"ROE: {profitability.get('roe', 0):.1%}"
            })
        
        # Strong liquidity
        liquidity = financial_metrics.get('liquidity', {})
        if liquidity.get('current_ratio', 0) > 3.0:
            strengths.append({
                'strength': 'Strong Liquidity Position',
                'description': 'High current ratio provides substantial financial flexibility',
                'metric': f"Current Ratio: {liquidity.get('current_ratio', 0):.1f}"
            })
        
        # Low debt
        leverage = financial_metrics.get('leverage', {})
        if leverage.get('debt_to_equity', 1.0) < 0.3:
            strengths.append({
                'strength': 'Conservative Debt Management',
                'description': 'Low debt-to-equity ratio indicates conservative financial management',
                'metric': f"Debt-to-Equity: {leverage.get('debt_to_equity', 0):.1f}"
            })
        
        return strengths
    
    def _generate_future_outlook(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate future financial outlook"""
        outlook = {
            'outlook': 'neutral',
            'key_factors': [],
            'growth_prospects': 'moderate',
            'risk_factors': [],
            'recommendations': []
        }
        
        health_score = results.get('financial_health_score', 50)
        trend_analysis = results.get('trend_analysis', {})
        growth_metrics = results.get('financial_metrics', {}).get('growth', {})
        
        # Determine overall outlook
        if health_score > 75 and trend_analysis.get('revenue_trend') in ['improving', 'stable']:
            outlook['outlook'] = 'positive'
            outlook['key_factors'].append('Strong financial health metrics')
        elif health_score < 40 or trend_analysis.get('revenue_trend') == 'declining':
            outlook['outlook'] = 'negative'
            outlook['key_factors'].append('Weak financial performance indicators')
        
        # Growth prospects
        revenue_growth = growth_metrics.get('revenue_growth', 0)
        if revenue_growth > 0.15:
            outlook['growth_prospects'] = 'strong'
        elif revenue_growth < 0.05:
            outlook['growth_prospects'] = 'limited'
        
        # Recommendations
        red_flags = results.get('red_flags', [])
        if red_flags:
            outlook['recommendations'].append('Address identified financial red flags')
        
        if health_score > 70:
            outlook['recommendations'].append('Consider expansion opportunities given strong financial position')
        
        return outlook
    
    def _calculate_confidence_scores(self, financial_data: Dict[str, Any], 
                                   results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate confidence scores for the analysis"""
        confidence_scores = {
            'data_completeness': 0.0,
            'data_quality': 0.0,
            'temporal_coverage': 0.0,
            'analysis_reliability': 0.0,
            'overall_confidence': 0.0
        }
        
        # Data completeness score
        key_metrics = financial_data.get('key_metrics', {})
        required_metrics = ['revenue', 'net_income', 'total_assets', 'total_equity']
        available_metrics = sum(1 for metric in required_metrics if metric in key_metrics)
        confidence_scores['data_completeness'] = available_metrics / len(required_metrics)
        
        # Data quality score (based on number of periods)
        periods = financial_data.get('periods', [])
        confidence_scores['data_quality'] = min(len(periods) / 5.0, 1.0)  # Max at 5 periods
        
        # Temporal coverage score
        if len(periods) >= 3:
            confidence_scores['temporal_coverage'] = 1.0
        elif len(periods) >= 2:
            confidence_scores['temporal_coverage'] = 0.7
        else:
            confidence_scores['temporal_coverage'] = 0.3
        
        # Analysis reliability (based on available calculations)
        calculated_metrics = sum(1 for cat in results['financial_metrics'].values() 
                               if isinstance(cat, dict) and cat)
        confidence_scores['analysis_reliability'] = min(calculated_metrics / 5.0, 1.0)
        
        # Overall confidence
        confidence_scores['overall_confidence'] = np.mean([
            confidence_scores['data_completeness'],
            confidence_scores['data_quality'], 
            confidence_scores['temporal_coverage'],
            confidence_scores['analysis_reliability']
        ])
        
        return confidence_scores