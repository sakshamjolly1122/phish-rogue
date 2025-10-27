"""
URL augmentation techniques for adversarial robustness.
"""
import random
import urllib.parse
from typing import List, Optional

class URLAugmenter:
    """URL augmentation for adversarial robustness."""
    
    def __init__(self, augment_prob: float = 0.2):
        self.augment_prob = augment_prob
        
        # Homoglyph mappings
        self.homoglyphs = {
            'a': ['@', 'а', 'α'],
            'e': ['3', 'е', 'ε'],
            'i': ['1', 'і', 'ι', '|'],
            'o': ['0', 'о', 'ο'],
            'u': ['μ', 'υ'],
            's': ['$', 'ѕ', 'ς'],
            'l': ['1', '|', 'ι'],
            't': ['+', 'τ'],
            'g': ['9', 'ɡ'],
            'b': ['6', 'ь'],
            'p': ['ρ'],
            'c': ['с', 'ς'],
            'n': ['η', 'п'],
            'm': ['м', 'μ'],
            'w': ['ω', 'ш'],
            'v': ['ν', 'ѵ'],
            'x': ['х', 'χ'],
            'y': ['у', 'γ'],
            'z': ['z', 'ζ']
        }
        
        # Fake subdomains
        self.fake_subdomains = [
            'www', 'secure', 'login', 'account', 'bank', 'paypal', 'amazon',
            'google', 'facebook', 'twitter', 'instagram', 'linkedin',
            'support', 'help', 'service', 'portal', 'app', 'api'
        ]
        
        # Zero-width characters
        self.zero_width_chars = ['\u200b', '\u200c', '\u200d', '\u2060']
    
    def homoglyph_swap(self, url: str) -> str:
        """Replace characters with homoglyphs."""
        result = []
        for char in url.lower():
            if char in self.homoglyphs and random.random() < 0.1:
                result.append(random.choice(self.homoglyphs[char]))
            else:
                result.append(char)
        return ''.join(result)
    
    def url_encoding(self, url: str) -> str:
        """Apply URL encoding to some characters."""
        result = []
        for char in url:
            if char in ':/?#[]@!$&\'()*+,;=' and random.random() < 0.05:
                result.append(urllib.parse.quote(char))
            else:
                result.append(char)
        return ''.join(result)
    
    def fake_subdomain(self, url: str) -> str:
        """Add fake subdomain."""
        if '://' in url:
            protocol, rest = url.split('://', 1)
            if '/' in rest:
                domain, path = rest.split('/', 1)
                fake_sub = random.choice(self.fake_subdomains)
                return f"{protocol}://{fake_sub}.{domain}/{path}"
        return url
    
    def zero_width_injection(self, url: str) -> str:
        """Inject zero-width characters."""
        result = []
        for char in url:
            result.append(char)
            if random.random() < 0.02:
                result.append(random.choice(self.zero_width_chars))
        return ''.join(result)
    
    def case_mixing(self, url: str) -> str:
        """Mix character cases."""
        result = []
        for char in url:
            if char.isalpha():
                if random.random() < 0.1:
                    result.append(char.swapcase())
                else:
                    result.append(char)
            else:
                result.append(char)
        return ''.join(result)
    
    def augment(self, url: str) -> str:
        """Apply random augmentation to URL."""
        if random.random() > self.augment_prob:
            return url
        
        # Choose random augmentation technique
        techniques = [
            self.homoglyph_swap,
            self.url_encoding,
            self.fake_subdomain,
            self.zero_width_injection,
            self.case_mixing
        ]
        
        technique = random.choice(techniques)
        return technique(url)
    
    def batch_augment(self, urls: List[str]) -> List[str]:
        """Apply augmentation to batch of URLs."""
        return [self.augment(url) for url in urls]
