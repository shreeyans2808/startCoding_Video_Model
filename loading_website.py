import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import logging
import pickle
from pathlib import Path
import re
import mimetypes
from datetime import datetime
import hashlib
import uuid
from PIL import Image
import tempfile
import io
from utils import *
from config import *
from add_to_index import add_content

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebsiteIndexer:
    def __init__(self, base_url):
        self.base_url = base_url
        self.domain = urlparse(base_url).netloc
        self.visited_urls = set()
        self.company_name = self._extract_company_name()
        self.index, self.metadata = load_or_create_index()
        
    def _extract_company_name(self):
        domain = self.domain.replace('www.', '')
        domain = re.sub(r'\.(com|org|net|edu|gov|io)$', '', domain)
        return domain.replace('-', ' ').replace('_', ' ').title()
    
    def _is_same_domain(self, url):
        parsed_url = urlparse(url)
        return parsed_url.netloc == self.domain
    
    def _normalize_url(self, url):
        parsed = urlparse(url)
        normalized = parsed._replace(fragment='').geturl()
        return normalized.rstrip('/')
    
    def _extract_text_content(self, soup):
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator=' ', strip=True)
        return re.sub(r'\s+', ' ', text).strip()
    
    def _extract_media_urls(self, soup, media_type):
        urls = []
        if media_type == 'image':
            for img in soup.find_all('img'):
                if img.get('src'):
                    urls.append(img['src'])
        elif media_type == 'video':
            for video in soup.find_all(['video', 'source']):
                if video.get('src'):
                    urls.append(video['src'])
        elif media_type == 'audio':
            for audio in soup.find_all(['audio', 'source']):
                if audio.get('src'):
                    urls.append(audio['src'])
        return urls
    
    def _download_media(self, url, media_type):
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '')
                if media_type in content_type:
                    return response.content
        except Exception as e:
            logger.error(f"Failed to download media from {url}: {str(e)}")
        return None
    
    def _process_page(self, url):
        if url in self.visited_urls:
            return
        
        self.visited_urls.add(url)
        logger.info(f"Processing: {url}")
        
        try:
            response = requests.get(url)
            if response.status_code != 200:
                logger.warning(f"Failed to fetch {url}: Status code {response.status_code}")
                return
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            text_content = self._extract_text_content(soup)
            if text_content:
                add_content(text_content, "text", url)
            
            for media_type in ['image', 'video', 'audio']:
                media_urls = self._extract_media_urls(soup, media_type)
                for media_url in media_urls:
                    absolute_url = urljoin(url, media_url)
                    media_content = self._download_media(absolute_url, media_type)
                    if media_content:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{media_type}') as temp_file:
                            temp_file.write(media_content)
                            temp_path = temp_file.name
                        
                        try:
                            add_content(temp_path, media_type, absolute_url)
                        finally:
                            os.unlink(temp_path)
            
            for link in soup.find_all('a'):
                href = link.get('href')
                if href:
                    absolute_url = urljoin(url, href)
                    normalized_url = self._normalize_url(absolute_url)
                    if self._is_same_domain(normalized_url) and normalized_url not in self.visited_urls:
                        self._process_page(normalized_url)
                        
        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")
    
    def crawl_and_index(self):
        logger.info(f"Starting to crawl {self.base_url}")
        self._process_page(self.base_url)
        index_filename = f"{self.company_name.lower().replace(' ', '_')}_index.bin"
        metadata_filename = f"{self.company_name.lower().replace(' ', '_')}_metadata.pkl"
        global FAISS_INDEX_PATH, METADATA_PATH
        FAISS_INDEX_PATH = os.path.join(INDEX_DIR, index_filename)
        METADATA_PATH = os.path.join(INDEX_DIR, metadata_filename)
        
        save_index(self.index, self.metadata)
        logger.info(f"Indexing complete. Index saved to {FAISS_INDEX_PATH}")
        return FAISS_INDEX_PATH

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        website_url = sys.argv[1]
    else:
        website_url = input("Please enter the website URL to crawl: ").strip()
    
    if not website_url.startswith(('http://', 'https://')):
        website_url = 'https://' + website_url
    
    try:
        indexer = WebsiteIndexer(website_url)
        index_file = indexer.crawl_and_index()
        print(f"Indexing complete. Index saved to {index_file}")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
