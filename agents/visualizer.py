from typing import Dict, Any, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import pandas as pd
import os
import openai
import requests
import json
import re
from datetime import datetime, timedelta
from dotenv import load_dotenv
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

class RealDataVisualizationAgent:
    def __init__(self, search_agent=None):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.search_agent = search_agent
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Real data sources
        self.arxiv_base_url = "http://export.arxiv.org/api/query"
        self.semantic_scholar_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        self.crossref_url = "https://api.crossref.org/works"
        self.pubmed_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        
        # Data extraction patterns
        self.number_patterns = {
            'percentage': r'(\d+(?:\.\d+)?)\s*%',
            'decimal': r'\b(\d+\.\d+)\b',
            'integer': r'\b(\d+)\b(?!\.)',
            'year': r'\b(19|20)\d{2}\b',
            'large_number': r'\b(\d{1,3}(?:,\d{3})+)\b',
            'scientific': r'(\d+(?:\.\d+)?)[eE]([+-]?\d+)',
            'ratio': r'(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)',
            'range': r'(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)'
        }
    
    def execute(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute real data visualization based on research"""
        
        try:
            print(f" Starting research-based visualization for: {task}")
            
            # Step 1: Extract specific research queries
            research_queries = self._extract_targeted_queries(task)
            # print(f" Generated {len(research_queries)} research queries")
            
            # Step 2: Gather real research data
            research_data = self._fetch_real_research_data(research_queries, task)
            # print(f" Collected data from {len(research_data.get('papers', []))} sources")
            
            # Step 3: Extract quantitative data from research
            extracted_data = self._extract_quantitative_data(research_data, task)
            # print(f" Extracted {sum(len(v) for v in extracted_data.values() if isinstance(v, list))} data points")
            
            # Step 4: Create research-informed visualizations
            visualizations = self._create_data_driven_visualizations(
                task, extracted_data, research_data
            )
            
            return {
                "visualizations": visualizations,
                "research_summary": self._generate_research_summary(research_data),
                "data_points_extracted": sum(len(v) for v in extracted_data.values() if isinstance(v, list)),
                "research_sources": len(research_data.get('papers', [])),
                "status": "completed"
            }
            
        except Exception as e:
            print(f" Error in visualization creation: {e}")
            return {
                "error": str(e),
                "status": "failed",
                "visualizations": []
            }
    
    def _extract_targeted_queries(self, task: str) -> List[str]:
        """Extract specific, targeted research queries using LLM"""
        
        prompt = f"""
        Task: {task}
        
        Generate 3-5 SPECIFIC research queries that will return papers with quantitative data.
        Focus on:
        - Exact statistics, benchmarks, survey results
        - Performance comparisons with numbers
        - Market analysis with figures
        - Scientific measurements and results
        - Trend analysis with temporal data
        
        Return ONLY a JSON array of specific search terms that researchers would use.
        Example: ["quantum computing error rates 2024", "post-quantum cryptography performance benchmarks"]
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            
            content = response.choices[0].message.content
            # Extract JSON array
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                queries = json.loads(json_match.group())
                return queries if isinstance(queries, list) else [content]
            
        except Exception as e:
            print(f"⚠️ LLM query extraction failed: {e}")
        
        # Fallback to keyword-based extraction
        return self._generate_domain_queries(task)
    
    def _generate_domain_queries(self, task: str) -> List[str]:
        """Generate domain-specific queries based on task content"""
        
        task_lower = task.lower()
        queries = []
        
        # Quantum computing domain
        if any(word in task_lower for word in ['quantum', 'qubit', 'cryptography']):
            queries.extend([
                "quantum computing error rates statistics",
                "quantum supremacy timeline benchmarks",
                "post-quantum cryptography adoption survey",
                "quantum algorithm performance comparison",
                "quantum cybersecurity threat assessment"
            ])
        
        # Cybersecurity domain
        elif any(word in task_lower for word in ['cyber', 'security', 'threat', 'vulnerability']):
            queries.extend([
                "cybersecurity incident statistics 2024",
                "data breach cost analysis report",
                "vulnerability disclosure timeline",
                "cyber attack frequency by sector",
                "security investment ROI metrics"
            ])
        
        # AI/ML domain
        elif any(word in task_lower for word in ['ai', 'machine learning', 'artificial intelligence']):
            queries.extend([
                "AI adoption rates by industry 2024",
                "machine learning model performance benchmarks",
                "AI investment trends venture capital",
                "neural network accuracy comparisons",
                "AI ethics compliance statistics"
            ])
        
        # Default technology queries
        else:
            queries.extend([
                f"{task_lower} statistics research",
                f"{task_lower} performance benchmarks",
                f"{task_lower} market analysis data",
                f"{task_lower} comparative study results"
            ])
        
        return queries[:5]  # Limit to 5 queries
    
    def _create_statistical_distribution(self, extracted_data: Dict[str, List], 
                                       research_data: Dict[str, Any], task: str) -> Dict[str, Any]:
        """Create statistical distribution analysis from real data"""
        
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            measurements = extracted_data.get('measurements', [])
            percentages = extracted_data.get('percentages', [])
            
            # Distribution histogram
            if measurements:
                n, bins, patches = ax1.hist(measurements, bins=25, alpha=0.7, color='#8AC926', edgecolor='black')
                # Color bars based on value
                for i, p in enumerate(patches):
                    if i < len(patches) // 3:
                        p.set_facecolor('#8AC926')
                    elif i < 2 * len(patches) // 3:
                        p.set_facecolor('#FFCA3A')
                    else:
                        p.set_facecolor('#FF595E')
                
                ax1.set_title('Measurement Distribution', fontsize=12, fontweight='bold')
                ax1.set_xlabel('Measurement Value')
                ax1.set_ylabel('Frequency')
                ax1.axvline(np.mean(measurements), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(measurements):.2f}')
                ax1.legend()
            
            # Q-Q plot for normality check
            if measurements and len(measurements) > 10:
                try:
                    from scipy import stats
                    stats.probplot(measurements, dist="norm", plot=ax2)
                    ax2.set_title('Q-Q Plot (Normality Check)', fontsize=12, fontweight='bold')
                except ImportError:
                    # Fallback without scipy
                    ax2.scatter(range(len(measurements)), sorted(measurements), alpha=0.6)
                    ax2.set_title('Sorted Values Plot', fontsize=12, fontweight='bold')
                    ax2.set_xlabel('Index')
                    ax2.set_ylabel('Value')
            
            # Percentage analysis
            if percentages:
                ax3.hist(percentages, bins=20, alpha=0.7, color='#6F1D1B', edgecolor='black')
                ax3.set_title('Percentage Values Distribution', fontsize=12, fontweight='bold')
                ax3.set_xlabel('Percentage')
                ax3.set_ylabel('Frequency')
                ax3.axvline(np.mean(percentages), color='yellow', linestyle='--', 
                           label=f'Mean: {np.mean(percentages):.1f}%')
                ax3.legend()
            
            # Summary statistics
            if measurements or percentages:
                data_for_stats = measurements if measurements else percentages
                stats_text = [
                    f"Count: {len(data_for_stats)}",
                    f"Mean: {np.mean(data_for_stats):.2f}",
                    f"Std: {np.std(data_for_stats):.2f}",
                    f"Min: {np.min(data_for_stats):.2f}",
                    f"Max: {np.max(data_for_stats):.2f}",
                    f"Median: {np.median(data_for_stats):.2f}"
                ]
                
                ax4.text(0.1, 0.9, '\n'.join(stats_text), transform=ax4.transAxes, 
                        fontsize=12, verticalalignment='top', fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
                ax4.set_title('Statistical Summary', fontsize=12, fontweight='bold')
                ax4.axis('off')
            
            plt.suptitle(f'Statistical Distribution Analysis: {task[:50]}...', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Add data source annotation
            fig.text(0.02, 0.02, f'Statistical analysis of {len(data_for_stats)} data points from research', 
                    fontsize=10, style='italic')
            
            chart_path = 'real_statistical_distribution.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return {
                "type": "statistical_distribution",
                "title": "Research-Based Statistical Distribution",
                "file_path": chart_path,
                "description": f"Statistical analysis of {len(data_for_stats)} extracted data points",
                "key_insights": self._generate_statistical_insights(measurements, percentages),
                "data_source": f"Extracted from {len(research_data.get('papers', []))} research papers"
            }
            
        except Exception as e:
            print(f" Statistical distribution creation failed: {e}")
            return None
    
    def _create_comparative_analysis(self, extracted_data: Dict[str, List], 
                                   research_data: Dict[str, Any], task: str) -> Dict[str, Any]:
        """Create comparative analysis from extracted percentages and metrics"""
        
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            percentages = extracted_data.get('percentages', [])
            performance_metrics = extracted_data.get('performance_metrics', [])
            
            # Percentage comparison by categories
            if percentages:
                # Create categories based on percentage ranges
                categories = ['Low (0-25%)', 'Medium (26-50%)', 'High (51-75%)', 'Very High (76-100%)']
                category_counts = [
                    len([p for p in percentages if 0 <= p <= 25]),
                    len([p for p in percentages if 26 <= p <= 50]),
                    len([p for p in percentages if 51 <= p <= 75]),
                    len([p for p in percentages if 76 <= p <= 100])
                ]
                
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
                bars = ax1.bar(categories, category_counts, color=colors, alpha=0.8, edgecolor='black')
                ax1.set_title('Percentage Distribution by Categories', fontsize=12, fontweight='bold')
                ax1.set_ylabel('Count')
                ax1.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, count in zip(bars, category_counts):
                    if count > 0:
                        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                                str(count), ha='center', va='bottom', fontweight='bold')
            
            # Performance comparison
            if performance_metrics:
                sorted_metrics = sorted(performance_metrics, reverse=True)
                top_10 = sorted_metrics[:10] if len(sorted_metrics) >= 10 else sorted_metrics
                
                ax2.barh(range(len(top_10)), top_10, color='#FF8C42', alpha=0.8, edgecolor='black')
                ax2.set_title('Top Performance Metrics', fontsize=12, fontweight='bold')
                ax2.set_xlabel('Performance Value')
                ax2.set_ylabel('Rank')
                ax2.set_yticks(range(len(top_10)))
                ax2.set_yticklabels([f'#{i+1}' for i in range(len(top_10))])
            
            # Comparative scatter plot
            if percentages and performance_metrics:
                min_len = min(len(percentages), len(performance_metrics))
                if min_len > 5:
                    x_data = percentages[:min_len]
                    y_data = performance_metrics[:min_len]
                    
                    scatter = ax3.scatter(x_data, y_data, alpha=0.6, s=80, c=range(min_len), 
                                         cmap='viridis', edgecolors='black')
                    ax3.set_title('Percentages vs Performance Metrics', fontsize=12, fontweight='bold')
                    ax3.set_xlabel('Percentage Values')
                    ax3.set_ylabel('Performance Metrics')
                    plt.colorbar(scatter, ax=ax3, label='Data Point Index')
                    
                    # Add correlation coefficient if possible
                    try:
                        correlation = np.corrcoef(x_data, y_data)[0, 1]
                        ax3.text(0.05, 0.95, f'Correlation: {correlation:.2f}', 
                                transform=ax3.transAxes, fontweight='bold',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    except:
                        pass
            
            # Research source breakdown
            papers = research_data.get('papers', [])
            source_counts = Counter(paper.get('source', 'Unknown') for paper in papers)
            
            if source_counts:
                sources = list(source_counts.keys())
                counts = list(source_counts.values())
                
                wedges, texts, autotexts = ax4.pie(counts, labels=sources, autopct='%1.1f%%', 
                                                  startangle=90, colors=['#FF9999', '#66B2FF', '#99FF99'])
                ax4.set_title('Data Sources Distribution', fontsize=12, fontweight='bold')
                
                # Enhance text readability
                for autotext in autotexts:
                    autotext.set_color('black')
                    autotext.set_fontweight('bold')
            
            plt.suptitle(f'Comparative Analysis: {task[:50]}...', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Add data source annotation
            fig.text(0.02, 0.02, f'Comparative analysis from {len(papers)} research sources', 
                    fontsize=10, style='italic')
            
            chart_path = 'real_comparative_analysis.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return {
                "type": "comparative_analysis",
                "title": "Research-Based Comparative Analysis",
                "file_path": chart_path,
                "description": f"Comparative analysis of {len(percentages)} percentages and {len(performance_metrics)} metrics",
                "key_insights": self._generate_comparative_insights(percentages, performance_metrics),
                "data_source": f"Analysis from {len(papers)} research papers"
            }
            
        except Exception as e:
            print(f" Comparative analysis creation failed: {e}")
            return None
    
    def _create_research_landscape_analysis(self, research_data: Dict[str, Any], task: str) -> Dict[str, Any]:
        """Create research landscape overview visualization"""
        
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            papers = research_data.get('papers', [])
            
            # Publication year distribution
            years = []
            for paper in papers:
                try:
                    year = int(paper.get('published', '2024'))
                    if 1990 <= year <= 2025:
                        years.append(year)
                except:
                    continue
            
            if years:
                year_counts = Counter(years)
                sorted_years = sorted(year_counts.keys())
                counts = [year_counts[year] for year in sorted_years]
                
                ax1.plot(sorted_years, counts, marker='o', linewidth=3, markersize=8, color='#2E86AB')
                ax1.fill_between(sorted_years, counts, alpha=0.3, color='#2E86AB')
                ax1.set_title('Research Publication Timeline', fontsize=12, fontweight='bold')
                ax1.set_xlabel('Year')
                ax1.set_ylabel('Number of Publications')
                ax1.grid(True, alpha=0.3)
            
            # Citation analysis (if available)
            citations = [paper.get('citations', 0) for paper in papers if paper.get('citations')]
            if citations:
                ax2.hist(citations, bins=20, alpha=0.7, color='#F18F01', edgecolor='black')
                ax2.set_title('Citation Distribution', fontsize=12, fontweight='bold')
                ax2.set_xlabel('Citation Count')
                ax2.set_ylabel('Number of Papers')
                ax2.axvline(np.mean(citations), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(citations):.0f}')
                ax2.legend()
            
            # Research source distribution
            source_counts = Counter(paper.get('source', 'Unknown') for paper in papers)
            if source_counts:
                sources = list(source_counts.keys())
                counts = list(source_counts.values())
                
                bars = ax3.bar(sources, counts, color=['#8AC926', '#FF595E', '#FFCA3A'], 
                              alpha=0.8, edgecolor='black')
                ax3.set_title('Research Sources', fontsize=12, fontweight='bold')
                ax3.set_xlabel('Source')
                ax3.set_ylabel('Number of Papers')
                ax3.tick_params(axis='x', rotation=45)
                
                # Add value labels
                for bar, count in zip(bars, counts):
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            str(count), ha='center', va='bottom', fontweight='bold')
            
            # Research quality indicators
            quality_metrics = {
                'Total Papers': len(papers),
                'With Citations': len([p for p in papers if p.get('citations', 0) > 0]),
                'Recent (2020+)': len([p for p in papers if int(p.get('published', '0')) >= 2020]),
                'High Quality': len([p for p in papers if p.get('citations', 0) > 10])
            }
            
            metrics = list(quality_metrics.keys())
            values = list(quality_metrics.values())
            
            bars = ax4.barh(metrics, values, color='#6F1D1B', alpha=0.8, edgecolor='black')
            ax4.set_title('Research Quality Metrics', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Count')
            
            # Add value labels
            for bar, value in zip(bars, values):
                ax4.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                        str(value), ha='left', va='center', fontweight='bold')
            
            plt.suptitle(f'Research Landscape Analysis: {task[:50]}...', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Add comprehensive data source annotation
            total_citations = sum(citations) if citations else 0
            fig.text(0.02, 0.02, 
                    f'Analysis of {len(papers)} papers | Total Citations: {total_citations} | Years: {min(years, default=2024)}-{max(years, default=2024)}', 
                    fontsize=10, style='italic')
            
            chart_path = 'real_research_landscape.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return {
                "type": "research_landscape",
                "title": "Research Landscape Analysis",
                "file_path": chart_path,
                "description": f"Overview of {len(papers)} research papers and their characteristics",
                "key_insights": self._generate_landscape_insights(papers, years, citations),
                "data_source": f"Meta-analysis of {len(papers)} research papers"
            }
            
        except Exception as e:
            print(f" Research landscape analysis creation failed: {e}")
            return None
    
    def _create_comprehensive_overview(self, extracted_data: Dict[str, List], 
                                     research_data: Dict[str, Any], task: str) -> Dict[str, Any]:
        """Create comprehensive overview when specific visualizations aren't possible"""
        
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Data availability overview
            data_types = []
            data_counts = []
            for key, values in extracted_data.items():
                if isinstance(values, list) and len(values) > 0:
                    data_types.append(key.replace('_', ' ').title())
                    data_counts.append(len(values))
            
            if data_types:
                bars = ax1.bar(data_types, data_counts, color='#FF6B6B', alpha=0.8, edgecolor='black')
                ax1.set_title('Extracted Data Types', fontsize=12, fontweight='bold')
                ax1.set_ylabel('Count of Data Points')
                ax1.tick_params(axis='x', rotation=45)
                
                # Add value labels
                for bar, count in zip(bars, data_counts):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            str(count), ha='center', va='bottom', fontweight='bold')
            
            # Combined data visualization
            all_numeric_data = []
            for key, values in extracted_data.items():
                if isinstance(values, list) and values:
                    all_numeric_data.extend(values)
            
            if all_numeric_data:
                ax2.hist(all_numeric_data, bins=30, alpha=0.7, color='#4ECDC4', edgecolor='black')
                ax2.set_title('All Extracted Values Distribution', fontsize=12, fontweight='bold')
                ax2.set_xlabel('Value')
                ax2.set_ylabel('Frequency')
                ax2.axvline(np.mean(all_numeric_data), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(all_numeric_data):.2f}')
                ax2.legend()
            
            # Research timeline
            papers = research_data.get('papers', [])
            years = []
            for paper in papers:
                try:
                    year = int(paper.get('published', '2024'))
                    years.append(year)
                except:
                    continue
            
            if years:
                year_counts = Counter(years)
                sorted_years = sorted(year_counts.keys())
                counts = [year_counts[year] for year in sorted_years]
                
                ax3.plot(sorted_years, counts, marker='s', linewidth=3, markersize=8, color='#45B7D1')
                ax3.fill_between(sorted_years, counts, alpha=0.3, color='#45B7D1')
                ax3.set_title('Research Activity Timeline', fontsize=12, fontweight='bold')
                ax3.set_xlabel('Year')
                ax3.set_ylabel('Publications')
                ax3.grid(True, alpha=0.3)
            
            # Summary statistics
            summary_stats = {
                'Total Papers': len(papers),
                'Data Points': len(all_numeric_data),
                'Data Types': len(data_types),
                'Year Range': f"{min(years, default=2024)}-{max(years, default=2024)}" if years else "N/A"
            }
            
            stats_text = '\n'.join([f"{k}: {v}" for k, v in summary_stats.items()])
            ax4.text(0.1, 0.7, stats_text, transform=ax4.transAxes, fontsize=14, 
                    verticalalignment='top', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
            ax4.set_title('Research Summary', fontsize=12, fontweight='bold')
            ax4.axis('off')
            
            plt.suptitle(f'Comprehensive Research Overview: {task[:50]}...', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Add data source annotation
            fig.text(0.02, 0.02, 
                    f'Comprehensive analysis: {len(papers)} papers, {len(all_numeric_data)} data points', 
                    fontsize=10, style='italic')
            
            chart_path = 'real_comprehensive_overview.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return {
                "type": "comprehensive_overview",
                "title": "Comprehensive Research Overview",
                "file_path": chart_path,
                "description": f"Overview of {len(papers)} papers with {len(all_numeric_data)} extracted data points",
                "key_insights": self._generate_overview_insights(extracted_data, papers),
                "data_source": f"Comprehensive analysis of {len(papers)} research papers"
            }
            
        except Exception as e:
            print(f" Comprehensive overview creation failed: {e}")
            return None
        
    # Continuation of RealDataVisualizationAgent class - Utility methods and data processing

    def _fetch_real_research_data(self, queries: List[str], task: str) -> Dict[str, Any]:
        """Fetch real research data from multiple academic sources"""
        
        all_papers = []
        successful_queries = 0
        
        for query in queries:
            try:
                # print(f"🔍 Searching: {query}")
                
                # Search arXiv
                arxiv_papers = self._search_arxiv_real(query)
                all_papers.extend(arxiv_papers)
                
                # Search Semantic Scholar
                scholar_papers = self._search_semantic_scholar_real(query)
                all_papers.extend(scholar_papers)
                
                # Search CrossRef for more papers
                crossref_papers = self._search_crossref(query)
                all_papers.extend(crossref_papers)
                
                successful_queries += 1
                
            except Exception as e:
                print(f"⚠️ Query '{query}' failed: {e}")
                continue
        
        # Remove duplicates based on title similarity
        unique_papers = self._deduplicate_papers(all_papers)
        
        return {
            "papers": unique_papers,
            "total_queries": len(queries),
            "successful_queries": successful_queries,
            "raw_paper_count": len(all_papers),
            "unique_paper_count": len(unique_papers)
        }
    
    def _search_arxiv_real(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search arXiv and extract real paper data"""
        
        try:
            params = {
                'search_query': f'all:{query}',
                'start': 0,
                'max_results': max_results,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            
            response = requests.get(self.arxiv_base_url, params=params, timeout=15)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            papers = []
            
            # Define namespaces
            ns = {'atom': 'http://www.w3.org/2005/Atom',
                  'arxiv': 'http://arxiv.org/schemas/atom'}
            
            for entry in root.findall('atom:entry', ns):
                try:
                    title = entry.find('atom:title', ns)
                    summary = entry.find('atom:summary', ns)
                    published = entry.find('atom:published', ns)
                    
                    if title is not None and summary is not None:
                        paper = {
                            'title': title.text.strip().replace('\n', ' '),
                            'abstract': summary.text.strip().replace('\n', ' '),
                            'published': published.text[:4] if published is not None else '2024',
                            'source': 'arXiv',
                            'url': entry.find('atom:id', ns).text if entry.find('atom:id', ns) is not None else '',
                            'query': query
                        }
                        papers.append(paper)
                        
                except Exception as e:
                    print(f"⚠️ Error parsing arXiv entry: {e}")
                    continue
            
            # print(f" arXiv: Found {len(papers)} papers for '{query}'")
            return papers
            
        except Exception as e:
            # print(f" arXiv search failed for '{query}': {e}")
            return []
    
    def _search_semantic_scholar_real(self, query: str, limit: int = 10) -> List[Dict]:
        """Search Semantic Scholar for real research data"""
        
        try:
            params = {
                'query': query,
                'limit': limit,
                'fields': 'title,abstract,year,citationCount,authors,venue,url'
            }
            
            headers = {'User-Agent': 'Research-Visualization-Agent/1.0'}
            response = requests.get(
                self.semantic_scholar_url, 
                params=params, 
                headers=headers,
                timeout=15
            )
            response.raise_for_status()
            
            data = response.json()
            papers = []
            
            for paper in data.get('data', []):
                try:
                    if paper.get('title') and paper.get('abstract'):
                        papers.append({
                            'title': paper['title'],
                            'abstract': paper['abstract'],
                            'published': str(paper.get('year', 2024)),
                            'citations': paper.get('citationCount', 0),
                            'source': 'Semantic Scholar',
                            'venue': paper.get('venue', {}).get('name', '') if paper.get('venue') else '',
                            'url': paper.get('url', ''),
                            'query': query
                        })
                except Exception as e:
                    # print(f"⚠️ Error parsing Semantic Scholar entry: {e}")
                    continue
            
            # print(f" Semantic Scholar: Found {len(papers)} papers for '{query}'")
            return papers
            
        except Exception as e:
            # print(f" Semantic Scholar search failed for '{query}': {e}")
            return []
    
    def _search_crossref(self, query: str, rows: int = 5) -> List[Dict]:
        """Search CrossRef for additional research papers"""
        
        try:
            params = {
                'query': query,
                'rows': rows,
                'sort': 'relevance',
                'filter': 'type:journal-article'
            }
            
            response = requests.get(self.crossref_url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            papers = []
            
            for item in data.get('message', {}).get('items', []):
                try:
                    title = ' '.join(item.get('title', ['']))
                    abstract = ' '.join(item.get('abstract', ['']))
                    
                    if title and len(title) > 10:  # Basic quality check
                        papers.append({
                            'title': title,
                            'abstract': abstract,
                            'published': str(item.get('published-print', {}).get('date-parts', [[2024]])[0][0]),
                            'source': 'CrossRef',
                            'journal': item.get('container-title', [''])[0] if item.get('container-title') else '',
                            'doi': item.get('DOI', ''),
                            'query': query
                        })
                except Exception as e:
                    # print(f"⚠️ Error parsing CrossRef entry: {e}")
                    continue
            
            # print(f" CrossRef: Found {len(papers)} papers for '{query}'")
            return papers
            
        except Exception as e:
            # print(f" CrossRef search failed for '{query}': {e}")
            return []
    
    def _deduplicate_papers(self, papers: List[Dict]) -> List[Dict]:
        """Remove duplicate papers based on title similarity"""
        
        unique_papers = []
        seen_titles = set()
        
        for paper in papers:
            title = paper.get('title', '').lower().strip()
            # Create a simplified version for comparison
            title_simple = re.sub(r'[^\w\s]', '', title)
            title_words = set(title_simple.split())
            
            # Check if this is similar to any existing title
            is_duplicate = False
            for seen_title in seen_titles:
                seen_words = set(seen_title.split())
                # If more than 70% of words overlap, consider it a duplicate
                if len(title_words & seen_words) / max(len(title_words), len(seen_words), 1) > 0.7:
                    is_duplicate = True
                    break
            
            if not is_duplicate and len(title) > 10:
                unique_papers.append(paper)
                seen_titles.add(title_simple)
        
        return unique_papers
    
    def _extract_quantitative_data(self, research_data: Dict[str, Any], task: str) -> Dict[str, List]:
        """Extract real quantitative data from research papers"""
        
        papers = research_data.get('papers', [])
        extracted_data = {
            'percentages': [],
            'years': [],
            'measurements': [],
            'comparisons': [],
            'performance_metrics': [],
            'survey_results': [],
            'temporal_data': [],
            'citations': []
        }
        
        for paper in papers:
            try:
                # Extract from abstract and title
                text_content = f"{paper.get('title', '')} {paper.get('abstract', '')}"
                
                # Extract percentages
                percentages = re.findall(self.number_patterns['percentage'], text_content)
                extracted_data['percentages'].extend([float(p) for p in percentages if 0 <= float(p) <= 100])
                
                # Extract years
                years = re.findall(self.number_patterns['year'], text_content)
                extracted_data['years'].extend([int(y) for y in years if 1990 <= int(y) <= 2025])
                
                # Extract decimal measurements
                decimals = re.findall(self.number_patterns['decimal'], text_content)
                extracted_data['measurements'].extend([float(d) for d in decimals if 0 < float(d) < 1000])
                
                # Extract large numbers (potentially statistics)
                large_nums = re.findall(self.number_patterns['large_number'], text_content)
                for num_str in large_nums:
                    try:
                        num = int(num_str.replace(',', ''))
                        if 1000 <= num <= 1000000:  # Reasonable range for statistics
                            extracted_data['survey_results'].append(num)
                    except:
                        continue
                
                # Extract citation count if available
                if 'citations' in paper and isinstance(paper['citations'], int):
                    extracted_data['citations'].append(paper['citations'])
                
                # Extract publication year
                try:
                    pub_year = int(paper.get('published', '2024'))
                    if 2000 <= pub_year <= 2025:
                        extracted_data['temporal_data'].append(pub_year)
                except:
                    pass
                
                # Extract performance metrics (numbers followed by specific keywords)
                performance_keywords = ['accuracy', 'precision', 'recall', 'f1', 'score', 'rate', 'ratio', 'factor']
                for keyword in performance_keywords:
                    pattern = rf'(\d+(?:\.\d+)?)\s*{keyword}'
                    matches = re.findall(pattern, text_content.lower())
                    extracted_data['performance_metrics'].extend([float(m) for m in matches])
                
            except Exception as e:
                # print(f"⚠️ Data extraction error for paper: {e}")
                continue
        
        # Clean and validate extracted data
        for key, values in extracted_data.items():
            if isinstance(values, list):
                # Remove outliers and invalid values
                cleaned_values = [v for v in values if isinstance(v, (int, float)) and not np.isnan(v) and not np.isinf(v)]
                extracted_data[key] = cleaned_values
        
        return extracted_data
    
    def _create_data_driven_visualizations(self, task: str, extracted_data: Dict[str, List], 
                                         research_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create visualizations based on real extracted data"""
        
        visualizations = []
        
        # Determine which visualizations to create based on available data
        data_availability = {k: len(v) for k, v in extracted_data.items() if isinstance(v, list)}
        # print(f" Data availability: {data_availability}")
        
        # Create temporal analysis if we have years/temporal data
        if data_availability.get('temporal_data', 0) > 3:
            viz = self._create_temporal_analysis(extracted_data, research_data, task)
            if viz: visualizations.append(viz)
        
        # Create performance analysis if we have performance metrics
        if data_availability.get('performance_metrics', 0) > 5:
            viz = self._create_performance_analysis(extracted_data, research_data, task)
            if viz: visualizations.append(viz)
        
        # Create statistical distribution if we have measurements
        if data_availability.get('measurements', 0) > 10:
            viz = self._create_statistical_distribution(extracted_data, research_data, task)
            if viz: visualizations.append(viz)
        
        # Create comparative analysis if we have percentages
        if data_availability.get('percentages', 0) > 5:
            viz = self._create_comparative_analysis(extracted_data, research_data, task)
            if viz: visualizations.append(viz)
        
        # Create research landscape analysis
        viz = self._create_research_landscape_analysis(research_data, task)
        if viz: visualizations.append(viz)
        
        # If no specific visualizations were created, create a comprehensive overview
        if not visualizations:
            viz = self._create_comprehensive_overview(extracted_data, research_data, task)
            if viz: visualizations.append(viz)
        
        return visualizations
    
    def _create_temporal_analysis(self, extracted_data: Dict[str, List], 
                                research_data: Dict[str, Any], task: str) -> Dict[str, Any]:
        """Create temporal analysis visualization from real data"""
        
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Publication timeline
            years = extracted_data.get('temporal_data', [])
            year_counts = Counter(years)
            sorted_years = sorted(year_counts.keys())
            counts = [year_counts[year] for year in sorted_years]
            
            ax1.plot(sorted_years, counts, marker='o', linewidth=3, markersize=8, color='#2E86AB')
            ax1.fill_between(sorted_years, counts, alpha=0.3, color='#2E86AB')
            ax1.set_title('Research Publication Timeline', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Year')
            ax1.set_ylabel('Number of Publications')
            ax1.grid(True, alpha=0.3)
            
            # Trend analysis with percentages over time
            if extracted_data.get('percentages'):
                # Group percentages by year if we have enough data
                year_percentage_data = defaultdict(list)
                papers = research_data.get('papers', [])
                
                for paper in papers:
                    try:
                        year = int(paper.get('published', '2024'))
                        text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
                        percentages = re.findall(r'(\d+(?:\.\d+)?)\s*%', text)
                        for p in percentages:
                            if 0 <= float(p) <= 100:
                                year_percentage_data[year].append(float(p))
                    except:
                        continue
                
                if len(year_percentage_data) > 2:
                    years = sorted(year_percentage_data.keys())
                    avg_percentages = [np.mean(year_percentage_data[year]) for year in years]
                    
                    ax2.plot(years, avg_percentages, marker='s', linewidth=3, markersize=8, color='#A23B72')
                    ax2.set_title('Average Metrics Over Time', fontsize=14, fontweight='bold')
                    ax2.set_xlabel('Year')
                    ax2.set_ylabel('Average Percentage')
                    ax2.grid(True, alpha=0.3)
                else:
                    # Fallback: show distribution of all percentages
                    percentages = extracted_data.get('percentages', [])
                    ax2.hist(percentages, bins=15, alpha=0.7, color='#A23B72', edgecolor='black')
                    ax2.set_title('Distribution of Percentage Values', fontsize=14, fontweight='bold')
                    ax2.set_xlabel('Percentage Value')
                    ax2.set_ylabel('Frequency')
            
            plt.suptitle(f'Temporal Analysis: {task[:50]}...', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Add data source annotation
            fig.text(0.02, 0.02, f'Data from {len(research_data.get("papers", []))} research papers', 
                    fontsize=10, style='italic')
            
            chart_path = 'real_temporal_analysis.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return {
                "type": "temporal_analysis",
                "title": "Research-Based Temporal Analysis",
                "file_path": chart_path,
                "description": f"Temporal trends extracted from {len(years)} publication years",
                "key_insights": self._generate_temporal_insights(years, extracted_data),
                "data_source": f"{len(research_data.get('papers', []))} research papers"
            }
            
        except Exception as e:
            # print(f" Temporal analysis creation failed: {e}")
            return None
    
    def _create_performance_analysis(self, extracted_data: Dict[str, List], 
                                   research_data: Dict[str, Any], task: str) -> Dict[str, Any]:
        """Create performance analysis from real extracted metrics"""
        
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            performance_metrics = extracted_data.get('performance_metrics', [])
            measurements = extracted_data.get('measurements', [])
            
            # Performance distribution
            if performance_metrics:
                ax1.hist(performance_metrics, bins=20, alpha=0.7, color='#F18F01', edgecolor='black')
                ax1.set_title('Performance Metrics Distribution', fontsize=12, fontweight='bold')
                ax1.set_xlabel('Metric Value')
                ax1.set_ylabel('Frequency')
                ax1.axvline(np.mean(performance_metrics), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(performance_metrics):.2f}')
                ax1.legend()
            
            # Box plot for detailed analysis
            if performance_metrics and measurements:
                combined_data = [performance_metrics, measurements[:len(performance_metrics)]]
                ax2.boxplot(combined_data, labels=['Performance', 'Measurements'], patch_artist=True)
                ax2.set_title('Comparative Box Plot', fontsize=12, fontweight='bold')
                ax2.set_ylabel('Values')
            
            # Scatter plot if we have paired data
            if len(performance_metrics) > 5 and len(measurements) > 5:
                min_len = min(len(performance_metrics), len(measurements))
                x_data = performance_metrics[:min_len]
                y_data = measurements[:min_len]
                
                ax3.scatter(x_data, y_data, alpha=0.6, s=60, color='#C73E1D')
                ax3.set_title('Performance vs Measurements', fontsize=12, fontweight='bold')
                ax3.set_xlabel('Performance Metrics')
                ax3.set_ylabel('Measurements')
                
                # Add trend line
                if len(x_data) > 2:
                    z = np.polyfit(x_data, y_data, 1)
                    p = np.poly1d(z)
                    ax3.plot(x_data, p(x_data), "r--", alpha=0.8)
            
            # Top performers analysis
            if performance_metrics:
                top_performers = sorted(performance_metrics, reverse=True)[:10]
                ax4.bar(range(len(top_performers)), top_performers, color='#3F7CAC', alpha=0.8)
                ax4.set_title('Top Performance Values', fontsize=12, fontweight='bold')
                ax4.set_xlabel('Rank')
                ax4.set_ylabel('Performance Value')
            
            plt.suptitle(f'Performance Analysis: {task[:50]}...', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Add data source annotation
            fig.text(0.02, 0.02, f'Extracted {len(performance_metrics)} performance metrics from research', 
                    fontsize=10, style='italic')
            
            chart_path = 'real_performance_analysis.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return {
                "type": "performance_analysis",
                "title": "Research-Based Performance Analysis",
                "file_path": chart_path,
                "description": f"Analysis of {len(performance_metrics)} performance metrics from research",
                "key_insights": self._generate_performance_insights(performance_metrics, measurements),
                "data_source": f"Extracted from {len(research_data.get('papers', []))} research papers"
            }
            
        except Exception as e:
            # print(f" Performance analysis creation failed: {e}")
            return None
    
    # Insight generation methods
    def _generate_temporal_insights(self, years: List[int], extracted_data: Dict[str, List]) -> List[str]:
        """Generate insights from temporal analysis"""
        insights = []
        
        if years:
            year_range = max(years) - min(years)
            insights.append(f"Research spans {year_range} years from {min(years)} to {max(years)}")
            
            year_counts = Counter(years)
            peak_year = max(year_counts, key=year_counts.get)
            insights.append(f"Peak research activity occurred in {peak_year} with {year_counts[peak_year]} publications")
            
            recent_years = [y for y in years if y >= 2020]
            insights.append(f"{len(recent_years)} out of {len(years)} publications are from 2020 onwards")
        
        return insights
    
    def _generate_performance_insights(self, performance_metrics: List[float], measurements: List[float]) -> List[str]:
        """Generate insights from performance analysis"""
        insights = []
        
        if performance_metrics:
            insights.append(f"Performance metrics range from {min(performance_metrics):.2f} to {max(performance_metrics):.2f}")
            insights.append(f"Average performance: {np.mean(performance_metrics):.2f} (±{np.std(performance_metrics):.2f})")
            
            top_quartile = np.percentile(performance_metrics, 75)
            high_performers = len([m for m in performance_metrics if m >= top_quartile])
            insights.append(f"{high_performers} metrics ({high_performers/len(performance_metrics)*100:.1f}%) are in the top quartile")
        
        return insights
    
    def _generate_statistical_insights(self, measurements: List[float], percentages: List[float]) -> List[str]:
        """Generate insights from statistical distribution"""
        insights = []
        
        if measurements:
            insights.append(f"Measurements show {np.std(measurements)/np.mean(measurements)*100:.1f}% coefficient of variation")
            
        if percentages:
            high_percentages = len([p for p in percentages if p > 50])
            insights.append(f"{high_percentages} out of {len(percentages)} percentages exceed 50%")
            
        return insights
    
    def _generate_comparative_insights(self, percentages: List[float], performance_metrics: List[float]) -> List[str]:
        """Generate insights from comparative analysis"""
        insights = []
        
        if percentages and performance_metrics:
            if len(percentages) == len(performance_metrics):
                try:
                    correlation = np.corrcoef(percentages, performance_metrics)[0, 1]
                    if abs(correlation) > 0.3:
                        insights.append(f"Moderate correlation ({correlation:.2f}) between percentages and performance")
                except:
                    pass
            
            insights.append(f"Data distribution: {len(percentages)} percentages, {len(performance_metrics)} performance metrics")
        
        return insights
    
    def _generate_landscape_insights(self, papers: List[Dict], years: List[int], citations: List[int]) -> List[str]:
        """Generate insights from research landscape"""
        insights = []
        
        insights.append(f"Research landscape includes {len(papers)} papers")
        
        if citations:
            high_impact = len([c for c in citations if c > 50])
            insights.append(f"{high_impact} papers ({high_impact/len(citations)*100:.1f}%) have >50 citations")
        
        if years:
            recent_research = len([y for y in years if y >= 2020])
            insights.append(f"{recent_research/len(years)*100:.1f}% of research is from recent years (2020+)")
        
        return insights
    
    def _generate_overview_insights(self, extracted_data: Dict[str, List], papers: List[Dict]) -> List[str]:
        """Generate insights from comprehensive overview"""
        insights = []
        
        total_data_points = sum(len(v) for v in extracted_data.values() if isinstance(v, list))
        insights.append(f"Extracted {total_data_points} quantitative data points from {len(papers)} papers")
        
        data_types = len([k for k, v in extracted_data.items() if isinstance(v, list) and len(v) > 0])
        insights.append(f"Analysis covers {data_types} different types of quantitative data")
        
        return insights
    
    def _generate_research_summary(self, research_data: Dict[str, Any]) -> str:
        """Generate a comprehensive research summary"""
        papers = research_data.get('papers', [])
        
        summary_parts = [
            f"Analyzed {len(papers)} research papers",
            f"from {research_data.get('successful_queries', 0)} successful search queries.",
            f"Data sources include {len(set([p.get('source', 'Unknown') for p in papers]))} academic databases.",
        ]
        
        if papers:
            years = [int(p.get('published', '2024')) for p in papers]
            year_range = f"{min(years)}-{max(years)}" if years else "N/A"
            summary_parts.append(f"Research spans the period {year_range}.")
            
            citations = [p.get('citations', 0) for p in papers if p.get('citations')]
            if citations:
                total_citations = sum(citations)
                summary_parts.append(f"Combined citation impact: {total_citations} citations.")
        
        return " ".join(summary_parts)

