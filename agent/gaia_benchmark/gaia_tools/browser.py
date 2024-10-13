# Shamelessly stolen from Microsoft Autogen team: thanks to them for this great resource!
# https://github.com/microsoft/autogen/blob/gaia_multiagent_v01_march_1st/autogen/browser_utils.py
import json
import os
import requests
import re
import io
import uuid
import mimetypes
import time
import pathlib
import pathvalidate
from urllib.parse import urljoin, urlparse, unquote, parse_qs
from urllib.request import url2pathname
from typing import Any, Dict, List, Optional, Union, Tuple
from .mdconvert import MarkdownConverter, UnsupportedFormatException, FileConversionException
import serpapi

from .cookies import COOKIES


class SimpleTextBrowser:
    """(In preview) An extremely simple text-based web browser comparable to Lynx. Suitable for Agentic use."""

    def __init__(
        self,
        start_page: Optional[str] = None,
        viewport_size: Optional[int] = 1024 * 8,
        downloads_folder: Optional[Union[str, None]] = None,
        serpapi_key: Optional[Union[str, None]] = None,
        request_kwargs: Optional[Union[Dict[str, Any], None]] = None,
    ):
        self.start_page: str = start_page if start_page else "about:blank"
        self.viewport_size = viewport_size  # Applies only to the standard uri types
        self.downloads_folder = downloads_folder if downloads_folder else "./dump/artifacts"
        self.history: List[Tuple[str, float]] = list()
        self.page_title: Optional[str] = None
        self.viewport_current_page = 0
        self.viewport_pages: List[Tuple[int, int]] = list()
        self.set_address(self.start_page)
        self.serpapi_key = 'dee7b7e3f40d3c7c94b371f25ab12b57545fd67c'
        self.request_kwargs = request_kwargs
        #self.request_kwargs["cookies"] = COOKIES
        self._mdconvert = MarkdownConverter()
        self._page_content: str = ""

        self._find_on_page_query: Union[str, None] = None
        self._find_on_page_last_result: Union[int, None] = None  # Location of the last result

    @property
    def address(self) -> str:
        """Return the address of the current page."""
        return self.history[-1][0]

    def set_address(self, uri_or_path: str, filter_year: Optional[int] = None) -> None:
        # TODO: Handle anchors
        self.history.append((uri_or_path, time.time()))

        # Handle special URIs
        if uri_or_path == "about:blank":
            self._set_page_content("")
        elif uri_or_path.startswith("google:"):
            self._serpapi_search(uri_or_path[len("google:"):].strip(), filter_year=filter_year)
        else:
            if (
                not uri_or_path.startswith("http:")
                and not uri_or_path.startswith("https:")
                and not uri_or_path.startswith("file:")
            ):
                if len(self.history) > 1:
                    prior_address = self.history[-2][0]
                    uri_or_path = urljoin(prior_address, uri_or_path)
                    # Update the address with the fully-qualified path
                    self.history[-1] = (uri_or_path, self.history[-1][1])
            self._fetch_page(uri_or_path)

        self.viewport_current_page = 0
        self.find_on_page_query = None
        self.find_on_page_viewport = None

    @property
    def viewport(self) -> str:
        """Return the content of the current viewport."""
        bounds = self.viewport_pages[self.viewport_current_page]
        return self.page_content[bounds[0] : bounds[1]]

    @property
    def page_content(self) -> str:
        """Return the full contents of the current page."""
        return self._page_content

    def _set_page_content(self, content: str) -> None:
        """Sets the text content of the current page."""
        self._page_content = content
        self._split_pages()
        if self.viewport_current_page >= len(self.viewport_pages):
            self.viewport_current_page = len(self.viewport_pages) - 1

    def page_down(self) -> None:
        self.viewport_current_page = min(self.viewport_current_page + 1, len(self.viewport_pages) - 1)

    def page_up(self) -> None:
        self.viewport_current_page = max(self.viewport_current_page - 1, 0)

    def find_on_page(self, query: str) -> Union[str, None]:
        """Searches for the query from the current viewport forward, looping back to the start if necessary."""

        # Did we get here via a previous find_on_page search with the same query?
        # If so, map to find_next
        if query == self._find_on_page_query and self.viewport_current_page == self._find_on_page_last_result:
            return self.find_next()

        # Ok it's a new search start from the current viewport
        self._find_on_page_query = query
        viewport_match = self._find_next_viewport(query, self.viewport_current_page)
        if viewport_match is None:
            self._find_on_page_last_result = None
            return None
        else:
            self.viewport_current_page = viewport_match
            self._find_on_page_last_result = viewport_match
            return self.viewport

    def find_next(self) -> None:
        """Scroll to the next viewport that matches the query"""

        if self._find_on_page_query is None:
            return None

        starting_viewport = self._find_on_page_last_result
        if starting_viewport is None:
            starting_viewport = 0
        else:
            starting_viewport += 1
            if starting_viewport >= len(self.viewport_pages):
                starting_viewport = 0

        viewport_match = self._find_next_viewport(self._find_on_page_query, starting_viewport)
        if viewport_match is None:
            self._find_on_page_last_result = None
            return None
        else:
            self.viewport_current_page = viewport_match
            self._find_on_page_last_result = viewport_match
            return self.viewport

    def _find_next_viewport(self, query: str, starting_viewport: int) -> Union[int, None]:
        """Search for matches between the starting viewport looping when reaching the end."""

        if query is None:
            return None

        # Normalize the query, and convert to a regular expression
        nquery = re.sub(r"\*", "__STAR__", query)
        nquery = " " + (" ".join(re.split(r"\W+", nquery))).strip() + " "
        nquery = nquery.replace(" __STAR__ ", "__STAR__ ")  # Merge isolated stars with prior word
        nquery = nquery.replace("__STAR__", ".*").lower()

        if nquery.strip() == "":
            return None

        idxs = list()
        idxs.extend(range(starting_viewport, len(self.viewport_pages)))
        idxs.extend(range(0, starting_viewport))

        for i in idxs:
            bounds = self.viewport_pages[i]
            content = self.page_content[bounds[0] : bounds[1]]

            # TODO: Remove markdown links and images
            ncontent = " " + (" ".join(re.split(r"\W+", content))).strip().lower() + " "
            if re.search(nquery, ncontent):
                return i

        return None

    def visit_page(self, path_or_uri: str, filter_year: Optional[int] = None) -> str:
        """Update the address, visit the page, and return the content of the viewport."""
        self.set_address(path_or_uri, filter_year=filter_year)
        return self.viewport

    def _split_pages(self) -> None:
        # Do not split search results
        if self.address.startswith("google:"):
            self.viewport_pages = [(0, len(self._page_content))]
            return

        # Handle empty pages
        if len(self._page_content) == 0:
            self.viewport_pages = [(0, 0)]
            return

        # Break the viewport into pages
        self.viewport_pages = []
        start_idx = 0
        while start_idx < len(self._page_content):
            end_idx = min(start_idx + self.viewport_size, len(self._page_content))  # type: ignore[operator]
            # Adjust to end on a space
            while end_idx < len(self._page_content) and self._page_content[end_idx - 1] not in [" ", "\t", "\r", "\n"]:
                end_idx += 1
            self.viewport_pages.append((start_idx, end_idx))
            start_idx = end_idx

    def _google_serper_api_results(
        self,  **kwargs: Any
    ) -> dict:
        headers = {
            "X-API-KEY": self.serpapi_key or "",
            "Content-Type": "application/json",
        }


        params = {
            **{key: value for key, value in kwargs.items() if value is not None},
        }
        response = requests.post(
            f"https://google.serper.dev/search", headers=headers, params=params
        )
        response.raise_for_status()
        search_results = response.json()
        return search_results


    def _serpapi_search(self, query: str, filter_year: Optional[int] = None) -> None:
        if self.serpapi_key is None:
            raise ValueError("Missing SerpAPI key.")
        
        # params = {
        #     "engine": "google",
        #     "q": query,
        #     "api_key": self.serpapi_key,
        # }
        params = { "q": query,'gl':  "us",'hl': "en"}
        if filter_year is not None:
            params["tbs"] = f"cdr:1,cd_min:01/01/{filter_year},cd_max:12/31/{filter_year}"


        results =  self._google_serper_api_results(**params)
        self.page_title = f"{query} - Search"
        if "organic" not in results.keys():
            raise Exception(f"'organic_results' key not found for query: '{query}'. Use a less restrictive query.")
        if len(results['organic']) == 0:
            year_filter_message = f" with filter year={filter_year}" if filter_year is not None else ""
            self._set_page_content(f"No results found for '{query}'{year_filter_message}. Try with a more general query, or remove the year filter.")
            return

        def _prev_visit(url):
            for i in range(len(self.history) - 1, -1, -1):
                if self.history[i][0] == url:
                    return f"You previously visited this page {round(time.time() - self.history[i][1])} seconds ago.\n"
            return ""

        web_snippets: List[str] = list()
        idx = 0
        if "organic" in results:
            for page in results["organic"]:
                idx += 1
                date_published = ""
                if "date" in page:
                    date_published = "\nDate published: " + page["date"]

                source = ""
                if "source" in page:
                    source = "\nSource: " + page["source"]

                snippet = ""
                if "snippet" in page:
                    snippet = "\n" + page["snippet"]

                redacted_version = f"{idx}. [{page['title']}]({page['link']}){date_published}{source}\n{_prev_visit(page['link'])}{snippet}"

                redacted_version = redacted_version.replace("Your browser can't play this video.", "")
                web_snippets.append(redacted_version)


        content = (
            f"A Google search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n"
            + "\n\n".join(web_snippets)
        )

        self._set_page_content(content)


    def _fetch_page(self, url: str, extract_path = False) -> None:
        download_path = ""
        try:
            if url.startswith("file://"):
                download_path = os.path.normcase(os.path.normpath(unquote(url[7:])))
                res = self._mdconvert.convert_local(download_path)
                self.page_title = res.title
                self._set_page_content(res.text_content)
            else:
                # Prepare the request parameters
                request_kwargs = self.request_kwargs.copy() if self.request_kwargs is not None else {}
                request_kwargs["stream"] = True

                # Send a HTTP request to the URL
                response = requests.get(url, **request_kwargs)
                response.raise_for_status()

                # If the HTTP request was successful
                content_type = response.headers.get("content-type", "")

                # Text or HTML
                if "text/" in content_type.lower():
                    res = self._mdconvert.convert_response(response)
                    self.page_title = res.title
                    self._set_page_content(res.text_content)
                # A download
                else:
                    # Try producing a safe filename
                    fname = None
                    download_path = None
                    try:
                        fname = pathvalidate.sanitize_filename(os.path.basename(urlparse(url).path)).strip()
                        download_path = os.path.abspath(os.path.join(self.downloads_folder, fname))

                        suffix = 0
                        while os.path.exists(download_path) and suffix < 1000:
                            suffix += 1
                            base, ext = os.path.splitext(fname)
                            new_fname = f"{base}__{suffix}{ext}"
                            download_path = os.path.abspath(os.path.join(self.downloads_folder, new_fname))

                    except NameError:
                        pass

                    # No suitable name, so make one
                    if fname is None:
                        extension = mimetypes.guess_extension(content_type)
                        if extension is None:
                            extension = ".download"
                        fname = str(uuid.uuid4()) + extension
                        download_path = os.path.abspath(os.path.join(self.downloads_folder, fname))

                    # Open a file for writing
                    with open(download_path, "wb") as fh:
                        for chunk in response.iter_content(chunk_size=512):
                            fh.write(chunk)

                    # Render it
                    local_uri = pathlib.Path(download_path).as_uri()
                    self.set_address(local_uri)
                    if extract_path:
                        return local_uri


        except UnsupportedFormatException as e:
            print(e)
            self.page_title = ("Download complete.",)
            self._set_page_content(f"# Download complete\n\nSaved file to '{download_path}'")
        except FileConversionException as e:
            print(e)
            self.page_title = ("Download complete.",)
            self._set_page_content(f"# Download complete\n\nSaved file to '{download_path}'")
        except FileNotFoundError:
            self.page_title = "Error 404"
            self._set_page_content(f"## Error 404\n\nFile not found: {download_path}")
        except requests.exceptions.RequestException as request_exception:
            try:
                self.page_title = f"Error {response.status_code}"

                # If the error was rendered in HTML we might as well render it
                content_type = response.headers.get("content-type", "")
                if content_type is not None and "text/html" in content_type.lower():
                    res = self._mdconvert.convert(response)
                    self.page_title = f"Error {response.status_code}"
                    self._set_page_content(f"## Error {response.status_code}\n\n{res.text_content}")
                else:
                    text = ""
                    for chunk in response.iter_content(chunk_size=512, decode_unicode=True):
                        text += chunk
                    self.page_title = f"Error {response.status_code}"
                    self._set_page_content(f"## Error {response.status_code}\n\n{text}")
            except NameError:
                self.page_title = f"Error"
                self._set_page_content(f"## Error\n\n{str(request_exception)}")
