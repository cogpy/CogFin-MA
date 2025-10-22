"""
Basic tests for CogFin-MA framework
"""

import unittest
import asyncio
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from cogfin_ma import CogFinMA
from opencog_framework.atomspace import AtomSpace
from opencog_framework.atoms import ConceptNode, PredicateNode, EvaluationLink
from opencog_framework.truth_values import TruthValue
from knowledge_base.knowledge_base import KnowledgeBase


class TestOpenCogFramework(unittest.TestCase):
    """Test OpenCog framework components"""
    
    def setUp(self):
        self.atomspace = AtomSpace()
    
    def test_atomspace_creation(self):
        """Test AtomSpace creation and basic operations"""
        self.assertEqual(self.atomspace.size(), 0)
        
        # Create concepts
        apple = self.atomspace.create_concept("Apple")
        fruit = self.atomspace.create_concept("Fruit")
        
        self.assertEqual(self.atomspace.size(), 2)
        self.assertIsInstance(apple, ConceptNode)
        self.assertIsInstance(fruit, ConceptNode)
    
    def test_truth_values(self):
        """Test truth value operations"""
        tv1 = TruthValue(0.8, 0.9)
        tv2 = TruthValue(0.6, 0.7)
        
        self.assertEqual(tv1.get_strength(), 0.8)
        self.assertEqual(tv1.get_confidence(), 0.9)
        self.assertTrue(tv1.is_true())
        self.assertFalse(tv1.is_false())
    
    def test_evaluation_links(self):
        """Test evaluation link creation"""
        company = self.atomspace.create_concept("TestCompany")
        has_revenue = self.atomspace.create_predicate("HasRevenue")
        revenue_value = self.atomspace.create_concept("1000000")
        
        evaluation = EvaluationLink(
            has_revenue, 
            [company, revenue_value],
            TruthValue(0.9, 0.8)
        )
        
        self.atomspace.add(evaluation)
        
        self.assertEqual(self.atomspace.size(), 4)  # 3 nodes + 1 link
        self.assertEqual(evaluation.get_predicate(), has_revenue)


class TestKnowledgeBase(unittest.TestCase):
    """Test knowledge base functionality"""
    
    def setUp(self):
        self.kb = KnowledgeBase()
    
    def test_knowledge_base_initialization(self):
        """Test knowledge base initialization"""
        self.assertIsNotNone(self.kb.atomspace)
        self.assertIsNotNone(self.kb.pattern_matcher)
        
        # Check if financial ontology was initialized
        stats = self.kb.get_statistics()
        self.assertGreater(stats['total_concepts'], 0)
    
    def test_add_financial_data(self):
        """Test adding financial data to knowledge base"""
        company = "TestCorp"
        metric = "ROE"
        value = 0.15
        period = "2023"
        
        self.kb.add_financial_data(company, metric, value, period)
        
        # Query the data back
        metrics = self.kb.query_company_metrics(company)
        self.assertGreater(len(metrics), 0)
    
    def test_add_sentiment_data(self):
        """Test adding sentiment data to knowledge base"""
        company = "TestCorp"
        sentiment = "positive"
        score = 0.8
        source = "test_agent"
        
        self.kb.add_sentiment_data(company, sentiment, score, source)
        
        # Query sentiment data
        sentiment_data = self.kb.query_sentiment_analysis(company)
        self.assertGreater(len(sentiment_data), 0)


class TestCogFinMAFramework(unittest.TestCase):
    """Test main CogFin-MA framework"""
    
    def setUp(self):
        self.config = {
            'agents': {
                'sentiment': {'model_name': 'yiyanghkust/finbert-tone'},
                'price_correlation': {'correlation_window_days': 5},
                'news': {'relevance_threshold': 0.6},
                'financial_health': {},
                'synthesis': {'coordination_timeout': 60}
            }
        }
    
    def test_framework_initialization(self):
        """Test framework initialization"""
        try:
            framework = CogFinMA(self.config)
            self.assertTrue(framework.is_initialized)
            self.assertEqual(len(framework.agents), 5)  # 5 agents
            
            # Test system statistics
            stats = framework.get_system_statistics()
            self.assertIn('framework_stats', stats)
            self.assertIn('atomspace_stats', stats)
            
        except ImportError as e:
            # Skip test if dependencies not available
            self.skipTest(f"Skipping due to missing dependencies: {e}")
    
    def test_sample_data_validation(self):
        """Test input data validation"""
        try:
            framework = CogFinMA(self.config)
            
            # Valid data
            valid_data = {
                'company_name': 'TestCorp',
                'annual_reports': [],
                'news_articles': []
            }
            
            # This should not raise an exception
            framework._validate_input_data(valid_data)
            
            # Invalid data (missing required field)
            invalid_data = {
                'annual_reports': []
            }
            
            with self.assertRaises(ValueError):
                framework._validate_input_data(invalid_data)
                
        except ImportError:
            self.skipTest("Skipping due to missing dependencies")


async def run_async_tests():
    """Run async integration tests"""
    print("Running async integration tests...")
    
    try:
        # Create sample data
        sample_data = {
            'company_name': 'TestCorp',
            'stock_ticker': 'TEST',
            'industry': 'Technology',
            'annual_reports': [
                {
                    'title': 'TestCorp Annual Report 2023',
                    'text': 'TestCorp showed strong performance with revenue growth.',
                    'date': '2023-12-31'
                }
            ],
            'news_articles': [
                {
                    'title': 'TestCorp Beats Earnings',
                    'content': 'Company reported strong quarterly results.',
                    'source': 'test.com',
                    'published_date': '2024-01-15'
                }
            ]
        }
        
        # Test framework with minimal config
        minimal_config = {
            'agents': {
                'sentiment': {'model_name': 'yiyanghkust/finbert-tone'},
            }
        }
        
        framework = CogFinMA(minimal_config)
        print("✓ Framework initialized for async test")
        
        # Test knowledge base operations
        company = "TestCorp"
        framework.knowledge_base.add_financial_data(company, "TestMetric", 100.0, "2023")
        
        insights = framework.get_company_insights(company)
        print("✓ Knowledge base operations successful")
        
        print("✓ All async tests passed")
        
    except ImportError as e:
        print(f"⚠ Skipping async tests due to missing dependencies: {e}")
    except Exception as e:
        print(f"✗ Async test failed: {e}")


def main():
    """Run all tests"""
    print("=" * 60)
    print("CogFin-MA Framework - Basic Tests")
    print("=" * 60)
    
    # Run unit tests
    print("\n1. Running Unit Tests...")
    unittest.main(verbosity=2, exit=False, argv=[''])
    
    # Run async tests
    print("\n2. Running Async Integration Tests...")
    asyncio.run(run_async_tests())
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()