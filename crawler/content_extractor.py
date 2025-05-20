"""
Content extraction functionality for the Manzil Chatbot crawler.
"""

import re
from bs4 import BeautifulSoup
from utils.logging_config import get_logger
from utils.error_handlers import ContentExtractionException, ErrorHandler

logger = get_logger("crawler.content_extractor")


class ContentExtractor:
    """Extracts relevant content from HTML pages."""

    def __init__(self):
        """Initialize the content extractor."""
        # Patterns for identifying main content areas
        self.content_patterns = {
            "id": re.compile(r"^(content|main|article)"),
            "class": re.compile(r"^(content|main|article|post)"),
            "tag_priority": ["article", "main", "div", "section"],
        }

        # Patterns for identifying navigation/boilerplate areas
        self.noise_patterns = {
            "id": re.compile(r"^(nav|menu|header|footer|sidebar|comment)"),
            "class": re.compile(r"^(nav|menu|header|footer|sidebar|comment)"),
        }

        # Islamic finance specific terms for content relevance scoring
        self.domain_terms = [
            "halal",
            "islamic",
            "shariah",
            "sharia",
            "murabaha",
            "sukuk",
            "riba",
            "zakat",
            "financing",
            "mortgage",
            "investment",
            "will",
            "wasiyyah",
        ]

    def extract_from_html(self, html_content, url=""):
        """
        Extract main content from HTML content.

        Args:
            html_content (str): The HTML content to extract from
            url (str): The URL of the page (for logging purposes)

        Returns:
            dict: Extracted content with metadata
        """
        with ErrorHandler(error_type="content_extraction", reraise=True):
            if not html_content:
                raise ContentExtractionException("Empty HTML content")

            try:
                soup = BeautifulSoup(html_content, "html.parser")

                # Extract title
                title = self._extract_title(soup)

                # Extract meta description
                description = self._extract_meta_description(soup)

                # Remove noise elements
                self._remove_noise(soup)

                # Extract main content
                main_content = self._extract_main_content(soup)

                # Extract structured content
                structured_content = self._extract_structured_content(soup)

                # Clean and normalize the text
                cleaned_content = self._clean_text(main_content)

                # Check if we have meaningful content
                if len(cleaned_content.split()) < 20:
                    logger.warning(
                        f"Extracted content from {url} is very short ({len(cleaned_content.split())} words)"
                    )

                result = {
                    "title": title,
                    "description": description,
                    "content": cleaned_content,
                    "structured_content": structured_content,
                    "content_length": len(cleaned_content.split()),
                    "domain_relevance": self._calculate_domain_relevance(
                        cleaned_content
                    ),
                }

                logger.debug(
                    f"Successfully extracted content from {url}: {len(cleaned_content.split())} words"
                )
                return result

            except Exception as e:
                logger.error(f"Error extracting content from {url}: {str(e)}")
                raise ContentExtractionException(f"Failed to extract content: {str(e)}")

    def _extract_title(self, soup):
        """Extract the title of the page."""
        title_tag = soup.title
        if title_tag and title_tag.string:
            return title_tag.string.strip()

        # Try h1 if no title tag
        h1_tag = soup.find("h1")
        if h1_tag and h1_tag.text:
            return h1_tag.text.strip()

        return "No Title"

    def _extract_meta_description(self, soup):
        """Extract meta description."""
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and "content" in meta_desc.attrs:
            return meta_desc["content"].strip()
        return ""

    def _remove_noise(self, soup):
        """Remove boilerplate and navigation elements."""
        # Remove script, style elements
        for element in soup(["script", "style", "iframe", "noscript"]):
            element.decompose()

        # Remove elements by id/class patterns
        for noise_type, pattern in self.noise_patterns.items():
            for element in soup.find_all(attrs={noise_type: pattern}):
                element.decompose()

        # Remove hidden elements
        for element in soup.find_all(style=re.compile(r"display:\s*none")):
            element.decompose()

    def _extract_main_content(self, soup):
        """Extract the main content area of the page."""
        # Try to find main content container
        main_element = None

        # Look for elements with main content IDs/classes
        for content_type, pattern in self.content_patterns.items():
            if content_type != "tag_priority":
                main_element = soup.find(attrs={content_type: pattern})
                if main_element:
                    break

        # If not found, try tag-based approach
        if not main_element:
            for tag in self.content_patterns["tag_priority"]:
                main_element = soup.find(tag)
                if main_element:
                    break

        # If still not found, use the body
        if not main_element:
            main_element = soup.body

        # If we have a main element, get its text
        if main_element:
            content = main_element.get_text(separator=" ", strip=True)
        else:
            # Fallback to all text
            content = soup.get_text(separator=" ", strip=True)

        return content

    def _extract_structured_content(self, soup):
        """Extract structured content like headers, lists, etc."""
        structured = {}

        # Extract headers
        headers = []
        for tag in ["h1", "h2", "h3", "h4"]:
            for header in soup.find_all(tag):
                if header.text.strip():
                    headers.append({"level": int(tag[1]), "text": header.text.strip()})
        structured["headers"] = headers

        # Extract lists
        lists = []
        for list_tag in soup.find_all(["ul", "ol"]):
            items = [
                li.text.strip() for li in list_tag.find_all("li") if li.text.strip()
            ]
            if items:
                lists.append({"type": list_tag.name, "items": items})
        structured["lists"] = lists

        # Extract tables
        tables = []
        for table in soup.find_all("table"):
            table_data = []
            headers = []

            # Try to find table headers
            th_elements = table.find_all("th")
            if th_elements:
                headers = [th.text.strip() for th in th_elements]

            # Process table rows
            for row in table.find_all("tr"):
                cells = row.find_all(["td", "th"])
                if cells:
                    row_data = [cell.text.strip() for cell in cells]
                    table_data.append(row_data)

            if table_data:
                tables.append({"headers": headers, "rows": table_data})
        structured["tables"] = tables

        return structured

    def _clean_text(self, text):
        """Clean and normalize text."""
        if not text:
            return ""

        # Convert multiple spaces, newlines to single space
        text = re.sub(r"\s+", " ", text)

        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s.,-?!:;()"\'%$€£₹]', "", text)

        # Remove extra whitespace around punctuation
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)

        # Fix spacing after punctuation
        text = re.sub(r"([.,!?;:])(\w)", r"\1 \2", text)

        return text.strip()

    def _calculate_domain_relevance(self, text):
        """Calculate relevance score for Islamic finance domain."""
        if not text:
            return 0

        text_lower = text.lower()
        word_count = len(text_lower.split())

        if word_count == 0:
            return 0

        domain_term_count = sum(1 for term in self.domain_terms if term in text_lower)

        # Calculate term frequency (terms per 100 words)
        return (domain_term_count / word_count) * 100
