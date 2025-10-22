"""
Basic usage example for CogFin-MA framework
"""

import asyncio
import json
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from cogfin_ma import CogFinMA


def create_sample_data(company: str) -> dict:
    """Create sample data for demonstration"""
    return {
        'company_name': company,
        'stock_ticker': 'AAPL',  # Example ticker
        'industry': 'Technology',
        'annual_reports': [
            {
                'title': f'{company} Annual Report 2023',
                'text': f"""
                {company} delivered strong financial performance in 2023 with revenue growth of 15%.
                Net income increased to $12.5 billion, representing a significant improvement from last year.
                The company maintained strong liquidity with cash and equivalents of $8.2 billion.
                Total assets reached $85 billion while total debt remained manageable at $28 billion.
                Operating margin improved to 18.5%, demonstrating operational efficiency.
                Return on equity increased to 22%, indicating effective capital utilization.
                """,
                'date': '2023-12-31',
                'period': '2023'
            }
        ],
        'news_articles': [
            {
                'title': f'{company} Announces Strong Q4 Earnings Beat',
                'content': f'{company} reported quarterly earnings that exceeded analyst expectations, driven by strong product demand and operational efficiency.',
                'source': 'reuters.com',
                'published_date': '2024-01-15',
                'url': 'https://example.com/news1'
            },
            {
                'title': f'{company} Launches Innovative Product Line',
                'content': f'The company unveiled a breakthrough product innovation that is expected to drive significant growth in the coming quarters.',
                'source': 'bloomberg.com', 
                'published_date': '2024-02-01',
                'url': 'https://example.com/news2'
            }
        ],
        'earnings_transcripts': [
            {
                'title': f'{company} Q4 2023 Earnings Call',
                'text': f"""
                CEO: We are pleased to report exceptional performance this quarter.
                Revenue grew 18% year-over-year, exceeding our guidance.
                Our strategic initiatives are paying off with improved margins.
                We remain optimistic about future growth prospects.
                CFO: Strong cash generation allows us to invest in innovation while returning capital to shareholders.
                """,
                'date': '2024-01-15'
            }
        ],
        'financial_statements': [
            {
                'period': '2023',
                'revenue': 85000000000,  # $85B
                'net_income': 12500000000,  # $12.5B
                'total_assets': 85000000000,  # $85B
                'total_equity': 45000000000,  # $45B
                'total_debt': 28000000000,   # $28B
                'cash': 8200000000,          # $8.2B
                'operating_cash_flow': 15000000000  # $15B
            }
        ]
    }


async def basic_example():
    """Run basic CogFin-MA analysis example"""
    print("=" * 60)
    print("CogFin-MA Framework - Basic Usage Example")
    print("=" * 60)
    
    # Initialize the framework
    print("\n1. Initializing CogFin-MA Framework...")
    
    config = {
        'agents': {
            'sentiment': {
                'model_name': 'yiyanghkust/finbert-tone',
                'max_length': 512
            },
            'price_correlation': {
                'correlation_window_days': 5,
                'min_correlation_threshold': 0.3
            },
            'news': {
                'relevance_threshold': 0.6
            },
            'financial_health': {
                'industry_benchmarks': {}
            },
            'synthesis': {
                'coordination_timeout': 300
            }
        }
    }
    
    try:
        framework = CogFinMA(config)
        print("✓ Framework initialized successfully")
        
        # Print system statistics
        stats = framework.get_system_statistics()
        print(f"  - Agents initialized: {stats['framework_stats']['agents_initialized']}")
        print(f"  - AtomSpace size: {stats['atomspace_stats']['total_atoms']}")
        
    except Exception as e:
        print(f"✗ Failed to initialize framework: {e}")
        return
    
    # Prepare sample data
    print("\n2. Preparing sample data...")
    company = "TechCorp Inc"
    data = create_sample_data(company)
    print(f"✓ Sample data prepared for {company}")
    print(f"  - Annual reports: {len(data['annual_reports'])}")
    print(f"  - News articles: {len(data['news_articles'])}")
    print(f"  - Earnings transcripts: {len(data['earnings_transcripts'])}")
    print(f"  - Financial statements: {len(data['financial_statements'])}")
    
    # Run comprehensive analysis
    print(f"\n3. Running comprehensive analysis for {company}...")
    
    try:
        start_time = datetime.now()
        results = await framework.analyze_company(company, data)
        end_time = datetime.now()
        
        analysis_duration = (end_time - start_time).total_seconds()
        print(f"✓ Analysis completed in {analysis_duration:.2f} seconds")
        
        # Display results summary
        print(f"\n4. Analysis Results Summary:")
        print(f"  - Analysis ID: {results['analysis_id']}")
        print(f"  - Status: {results['status']}")
        
        agent_results = results.get('agent_results', {})
        print(f"  - Agents executed: {len(agent_results)}")
        
        for agent_name, agent_result in agent_results.items():
            if 'error' in agent_result:
                print(f"    • {agent_name}: ✗ Failed ({agent_result['error']})")
            else:
                print(f"    • {agent_name}: ✓ Completed")
        
        # Show synthesis results
        synthesis_results = results.get('synthesis_results', {})
        if synthesis_results and 'error' not in synthesis_results:
            print(f"\n5. Synthesis Results:")
            
            # Try to extract key insights
            synthesis_data = synthesis_results.get('synthesis', {})
            if synthesis_data:
                overall_sentiment = synthesis_data.get('overall_sentiment', 'N/A')
                financial_health = synthesis_data.get('financial_health_score', 'N/A')
                market_impact = synthesis_data.get('market_impact_assessment', 'N/A')
                
                print(f"  - Overall Sentiment: {overall_sentiment}")
                print(f"  - Financial Health Score: {financial_health}")
                print(f"  - Market Impact Assessment: {market_impact}")
                
                investment_thesis = synthesis_data.get('investment_thesis', {})
                if investment_thesis:
                    recommendation = investment_thesis.get('recommendation', 'N/A')
                    conviction = investment_thesis.get('conviction_level', 'N/A')
                    print(f"  - Investment Recommendation: {recommendation} ({conviction} conviction)")
        
        # Show performance metrics
        performance = results.get('performance_metrics', {})
        if performance:
            print(f"\n6. Performance Metrics:")
            print(f"  - Success Rate: {performance.get('success_rate', 0):.1%}")
            print(f"  - Average Confidence: {performance.get('average_confidence', 0):.2f}")
            print(f"  - Successful Agents: {performance.get('successful_agents', 0)}")
            print(f"  - Failed Agents: {performance.get('failed_agents', 0)}")
        
    except Exception as e:
        print(f"✗ Analysis failed: {e}")
        return
    
    # Query knowledge base insights
    print(f"\n7. Querying Knowledge Base Insights...")
    
    try:
        insights = framework.get_company_insights(company)
        
        if 'error' not in insights:
            print(f"✓ Knowledge base insights retrieved")
            print(f"  - Financial Strength: {insights.get('financial_strength', 'N/A')}")
            print(f"  - Sentiment Trend: {insights.get('sentiment_trend', 'N/A')}")
            print(f"  - News Impact: {insights.get('news_impact', 'N/A')}")
            print(f"  - Overall Assessment: {insights.get('overall_assessment', 'N/A')}")
            print(f"  - Confidence: {insights.get('confidence', 0):.2f}")
            
            reasoning = insights.get('reasoning', [])
            if reasoning:
                print(f"  - Reasoning:")
                for reason in reasoning:
                    print(f"    • {reason}")
        else:
            print(f"✗ Failed to get insights: {insights['error']}")
            
    except Exception as e:
        print(f"✗ Failed to query knowledge base: {e}")
    
    # Show final system statistics
    print(f"\n8. Final System Statistics:")
    final_stats = framework.get_system_statistics()
    
    print(f"  - Total Analyses: {final_stats['framework_stats']['total_analyses']}")
    print(f"  - AtomSpace Atoms: {final_stats['atomspace_stats']['total_atoms']}")
    print(f"  - Knowledge Base Concepts: {final_stats['knowledge_base_stats']['total_concepts']}")
    print(f"  - Knowledge Base Evaluations: {final_stats['knowledge_base_stats']['total_evaluations']}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


async def batch_analysis_example():
    """Demonstrate batch analysis capability"""
    print("\n" + "=" * 60) 
    print("CogFin-MA Framework - Batch Analysis Example")
    print("=" * 60)
    
    framework = CogFinMA()
    
    companies = ["TechCorp Inc", "FinanceGiant Ltd", "HealthCare Solutions"]
    
    def data_provider(company: str) -> dict:
        """Provide data for each company"""
        return create_sample_data(company)
    
    print(f"\nRunning batch analysis for {len(companies)} companies...")
    print(f"Companies: {', '.join(companies)}")
    
    try:
        batch_results = await framework.batch_analyze(
            companies=companies,
            data_provider=data_provider,
            max_concurrent=2
        )
        
        print(f"\n✓ Batch analysis completed:")
        print(f"  - Total companies: {batch_results['total_companies']}")
        print(f"  - Completed: {batch_results['completed']}")
        print(f"  - Failed: {batch_results['failed']}")
        print(f"  - Success rate: {batch_results['success_rate']:.1%}")
        
        for company, result in batch_results['results'].items():
            status = "✓ Success" if 'error' not in result else f"✗ Failed ({result['error']})"
            print(f"  - {company}: {status}")
            
    except Exception as e:
        print(f"✗ Batch analysis failed: {e}")


if __name__ == "__main__":
    # Run basic example
    asyncio.run(basic_example())
    
    # Run batch analysis example  
    asyncio.run(batch_analysis_example())