"""
HTML content feature extraction for Stage-B.
"""
import requests
import time
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import re
from typing import Dict, List, Optional, Tuple
import numpy as np
from .utils import setup_logger, validate_url

class HTMLFeatureExtractor:
    """Extract lightweight HTML features for phishing detection."""
    
    def __init__(self, timeout_ms: int = 150, content_dim: int = 20):
        self.timeout_ms = timeout_ms
        self.content_dim = content_dim
        self.logger = setup_logger('html_features')
        
        # Feature names for interpretability
        self.feature_names = [
            'title_length',
            'meta_refresh_flag',
            'form_count',
            'script_count',
            'iframe_count',
            'onclick_count',
            'visible_text_ratio',
            'external_link_ratio',
            'html_length',
            'image_count',
            'input_count',
            'link_count',
            'div_count',
            'span_count',
            'table_count',
            'meta_count',
            'style_count',
            'javascript_count',
            'css_count',
            'suspicious_keywords'
        ]
    
    def fetch_html(self, url: str) -> Optional[Tuple[str, float]]:
        """Fetch HTML content with timeout."""
        try:
            start_time = time.time()
            response = requests.get(
                url,
                timeout=self.timeout_ms / 1000.0,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                },
                allow_redirects=True
            )
            fetch_time = time.time() - start_time
            
            if response.status_code == 200:
                return response.text, fetch_time
            else:
                return None, fetch_time
                
        except Exception as e:
            self.logger.debug(f"Failed to fetch {url}: {e}")
            return None, 0.0
    
    def extract_features(self, url: str) -> np.ndarray:
        """Extract HTML features from URL."""
        features = np.zeros(self.content_dim, dtype=np.float32)
        
        # Try to fetch HTML
        html_content, fetch_time = self.fetch_html(url)
        
        if html_content is None:
            # Return zero features if fetch failed
            return features
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Feature 0: Title length
            title_tag = soup.find('title')
            features[0] = len(title_tag.get_text().strip()) if title_tag else 0
            
            # Feature 1: Meta refresh flag
            meta_refresh = soup.find('meta', attrs={'http-equiv': re.compile(r'refresh', re.I)})
            features[1] = 1.0 if meta_refresh else 0.0
            
            # Feature 2: Form count
            features[2] = len(soup.find_all('form'))
            
            # Feature 3: Script count
            features[3] = len(soup.find_all('script'))
            
            # Feature 4: Iframe count
            features[4] = len(soup.find_all('iframe'))
            
            # Feature 5: Onclick count
            onclick_elements = soup.find_all(attrs={'onclick': True})
            features[5] = len(onclick_elements)
            
            # Feature 6: Visible text ratio
            all_text = soup.get_text()
            visible_text = re.sub(r'\s+', ' ', all_text).strip()
            html_length = len(html_content)
            features[6] = len(visible_text) / max(html_length, 1)
            
            # Feature 7: External link ratio
            links = soup.find_all('a', href=True)
            external_links = 0
            domain = urlparse(url).netloc
            
            for link in links:
                href = link['href']
                if href.startswith('http'):
                    link_domain = urlparse(href).netloc
                    if link_domain != domain:
                        external_links += 1
            
            features[7] = external_links / max(len(links), 1)
            
            # Feature 8: HTML length
            features[8] = min(len(html_content), 10000) / 1000.0  # Normalize
            
            # Feature 9: Image count
            features[9] = len(soup.find_all('img'))
            
            # Feature 10: Input count
            features[10] = len(soup.find_all('input'))
            
            # Feature 11: Link count
            features[11] = len(links)
            
            # Feature 12: Div count
            features[12] = len(soup.find_all('div'))
            
            # Feature 13: Span count
            features[13] = len(soup.find_all('span'))
            
            # Feature 14: Table count
            features[14] = len(soup.find_all('table'))
            
            # Feature 15: Meta count
            features[15] = len(soup.find_all('meta'))
            
            # Feature 16: Style count
            features[16] = len(soup.find_all('style'))
            
            # Feature 17: JavaScript count (inline scripts)
            script_tags = soup.find_all('script')
            js_count = sum(1 for script in script_tags if script.string and script.string.strip())
            features[17] = js_count
            
            # Feature 18: CSS count
            css_tags = soup.find_all('style')
            css_count = sum(1 for style in css_tags if style.string and style.string.strip())
            features[18] = css_count
            
            # Feature 19: Suspicious keywords
            suspicious_keywords = [
                'password', 'login', 'account', 'verify', 'confirm', 'update',
                'security', 'urgent', 'immediate', 'suspended', 'expired',
                'phishing', 'scam', 'fraud', 'click here', 'download now'
            ]
            
            text_content = soup.get_text().lower()
            keyword_count = sum(1 for keyword in suspicious_keywords if keyword in text_content)
            features[19] = min(keyword_count, 5) / 5.0  # Normalize to [0, 1]
            
        except Exception as e:
            self.logger.debug(f"Error parsing HTML for {url}: {e}")
        
        return features
    
    def batch_extract_features(self, urls: List[str]) -> np.ndarray:
        """Extract features for batch of URLs."""
        features_list = []
        
        for url in urls:
            if validate_url(url):
                features = self.extract_features(url)
                features_list.append(features)
            else:
                # Invalid URL - return zero features
                features_list.append(np.zeros(self.content_dim, dtype=np.float32))
        
        return np.array(features_list)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names for interpretability."""
        return self.feature_names.copy()

def extract_content_features(
    urls: List[str],
    timeout_ms: int = 150,
    content_dim: int = 20
) -> Tuple[np.ndarray, List[str]]:
    """Extract HTML content features for URLs."""
    extractor = HTMLFeatureExtractor(timeout_ms=timeout_ms, content_dim=content_dim)
    features = extractor.batch_extract_features(urls)
    feature_names = extractor.get_feature_names()
    
    return features, feature_names
