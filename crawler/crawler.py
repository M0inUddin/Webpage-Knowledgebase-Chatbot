"""
Web crawler for the Manzil Chatbot to extract content from the help center.
"""

import time
import requests
from urllib.parse import urljoin, urlparse
import urllib.robotparser
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup

from config import settings
from utils.logging_config import get_logger
from utils.error_handlers import CrawlerException, retry, ErrorHandler
from utils.rate_limiters import RateLimiter
from crawler.content_extractor import ContentExtractor

logger = get_logger("crawler.crawler")


class WebCrawler:
    """Crawler to extract content from Manzil's help center website."""

    def __init__(self):
        """Initialize the crawler with configuration settings."""
        self.base_url = settings.BASE_URL
        self.max_pages = settings.MAX_PAGES
        self.respect_robots = settings.RESPECT_ROBOTS_TXT
        self.crawl_delay = settings.CRAWL_DELAY
        self.user_agent = settings.USER_AGENT
        self.timeout = settings.REQUEST_TIMEOUT
        self.max_workers = settings.MAX_WORKERS
        self.rate_limit = settings.RATE_LIMIT

        # Initialize content extractor
        self.content_extractor = ContentExtractor()

        # Set up rate limiter
        self.rate_limiter = RateLimiter(
            default_rate=self.rate_limit / 60.0, default_capacity=10
        )

        # Initialize robots.txt parser
        self.robots_parser = None
        if self.respect_robots:
            self._setup_robots_parser()

        logger.info(
            f"WebCrawler initialized for {self.base_url} with max {self.max_pages} pages"
        )

    def _setup_robots_parser(self):
        """Setup robots.txt parser."""
        try:
            robots_url = urljoin(self.base_url, "/robots.txt")
            self.robots_parser = urllib.robotparser.RobotFileParser()
            self.robots_parser.set_url(robots_url)
            self.robots_parser.read()
            logger.info("Robots.txt parsed successfully")
        except Exception as e:
            logger.warning(f"Could not parse robots.txt: {str(e)}")
            self.robots_parser = None

    def _is_allowed_url(self, url):
        """Check if crawling a URL is allowed by robots.txt."""
        if not self.respect_robots or not self.robots_parser:
            return True

        try:
            return self.robots_parser.can_fetch(self.user_agent, url)
        except Exception as e:
            logger.warning(f"Error checking robots.txt for {url}: {str(e)}")
            return True

    def _is_valid_url(self, url):
        """Check if a URL is valid and within the target domain."""
        if not url:
            return False

        # Check if URL starts with base URL
        if not url.startswith(self.base_url):
            return False

        # Parse URL to check components
        parsed = urlparse(url)

        # Check scheme
        if parsed.scheme not in ["http", "https"]:
            return False

        # Check domain matches
        base_domain = urlparse(self.base_url).netloc
        if parsed.netloc != base_domain:
            return False

        # No fragment URLs (anchors)
        if parsed.fragment:
            return False

        # Exclude common non-content URLs
        excluded_patterns = [
            "/cdn-cgi/",
            "/wp-admin/",
            "/wp-content/",
            "/wp-includes/",
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".pdf",
            ".doc",
            ".docx",
            ".xls",
            ".xlsx",
            ".zip",
            ".tar",
            ".gz",
            ".exe",
        ]

        for pattern in excluded_patterns:
            if pattern in url:
                return False

        return True

    def _normalize_url(self, base_url, href):
        """Normalize URL to absolute form."""
        try:
            if not href:
                return None

            # Handle relative URLs
            if href.startswith("/"):
                return urljoin(base_url, href)

            # Handle absolute URLs
            if href.startswith(("http://", "https://")):
                return href

            # Handle other forms of relative URLs
            return urljoin(base_url, href)
        except Exception as e:
            logger.error(f"Error normalizing URL {href}: {str(e)}")
            return None

    @retry(
        exceptions=(requests.RequestException,),
        tries=3,
        delay=2,
        backoff=2,
        logger=logger,
    )
    def _fetch_url(self, url):
        """Fetch content from a URL with retries."""
        try:
            with self.rate_limiter.limited_context("crawler", 1):
                headers = {
                    "User-Agent": self.user_agent,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept-Encoding": "gzip, deflate, br",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1",
                }

                # Use a session for persistence across requests
                if not hasattr(self, "session"):
                    self.session = requests.Session()

                logger.debug(f"Fetching URL: {url}")
                response = self.session.get(url, headers=headers, timeout=self.timeout)
                response.raise_for_status()

                # Check content type
                content_type = response.headers.get("Content-Type", "")
                if "text/html" not in content_type.lower():
                    logger.warning(
                        f"Skipping non-HTML content: {url} (Content-Type: {content_type})"
                    )
                    return None, response.status_code

                return response.text, response.status_code
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            raise

    def _extract_links(self, html_content, base_url):
        """Extract links from HTML content."""
        links = []
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            for link in soup.find_all("a", href=True):
                href = link["href"]
                normalized_url = self._normalize_url(base_url, href)

                if normalized_url and self._is_valid_url(normalized_url):
                    links.append(normalized_url)
        except Exception as e:
            logger.error(f"Error extracting links from {base_url}: {str(e)}")

        return links

    def _process_url(self, url, visited):
        """Process a single URL and extract content."""
        if url in visited:
            return None

        if not self._is_allowed_url(url):
            logger.info(f"Skipping {url} as per robots.txt")
            return None

        try:
            html_content, status_code = self._fetch_url(url)

            if not html_content or status_code != 200:
                logger.warning(
                    f"Failed to retrieve content from {url}, status code: {status_code}"
                )
                return None

            # Extract content using the content extractor
            extracted_data = self.content_extractor.extract_from_html(html_content, url)

            # Add URL and timestamp
            extracted_data["url"] = url
            extracted_data["crawl_time"] = time.time()

            # Extract links for further crawling
            links = self._extract_links(html_content, url)

            return {"data": extracted_data, "links": links}
        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")
            return None

    def crawl(self):
        """
        Crawl the website and extract content.

        Returns:
            list: Extracted content with metadata
        """
        with ErrorHandler(error_type="web_crawling", reraise=True):
            visited = set()
            to_visit = [self.base_url]
            pages_content = []

            logger.info(
                f"Starting crawl of {self.base_url} with max pages limit: {self.max_pages}"
            )

            try:
                while to_visit and len(visited) < self.max_pages:
                    # Process URLs in batches with ThreadPoolExecutor
                    batch_size = min(
                        self.max_workers, len(to_visit), self.max_pages - len(visited)
                    )
                    current_batch = to_visit[:batch_size]
                    to_visit = to_visit[batch_size:]

                    logger.info(
                        f"Processing batch of {len(current_batch)} URLs, {len(visited)} visited so far"
                    )

                    with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                        # Submit all URLs for processing
                        future_results = {
                            executor.submit(self._process_url, url, visited): url
                            for url in current_batch
                        }

                        # Process results as they complete
                        for future in future_results:
                            url = future_results[future]
                            try:
                                result = future.result()
                                visited.add(url)

                                if result:
                                    pages_content.append(result["data"])

                                    # Add new links to visit
                                    for link in result["links"]:
                                        if (
                                            link not in visited
                                            and link not in to_visit
                                            and len(visited) + len(to_visit)
                                            < self.max_pages
                                        ):
                                            to_visit.append(link)
                            except Exception as e:
                                logger.error(
                                    f"Error processing future for {url}: {str(e)}"
                                )
                                visited.add(url)  # Mark as visited to avoid retrying

                    # Pause between batches to be polite
                    if to_visit and self.crawl_delay > 0:
                        time.sleep(self.crawl_delay)

                logger.info(
                    f"Crawl completed. Visited {len(visited)} pages, extracted content from {len(pages_content)} pages"
                )
                return pages_content
            except Exception as e:
                logger.error(f"Crawl failed with error: {str(e)}")
                raise CrawlerException(f"Failed to complete website crawl: {str(e)}")

    def crawl_single_url(self, url):
        """
        Crawl a single URL and extract content.

        Args:
            url (str): URL to crawl

        Returns:
            dict: Extracted content with metadata or None if failed
        """
        if not self._is_valid_url(url):
            logger.warning(f"Invalid URL: {url}")
            return None

        result = self._process_url(url, set())
        return result["data"] if result else None
