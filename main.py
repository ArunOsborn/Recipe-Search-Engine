# %%
from bs4 import BeautifulSoup

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *

stemmer = PorterStemmer()

import spacy
nlp = spacy.load("en_core_web_sm")

from math import log10 as log
import string
import re
import os
import sys
import hashlib

import time  # Just used for checking how long indexing takes
import json
import requests
from urllib.parse import urlparse, urljoin
import random

import matplotlib.pyplot as plt
import numpy as np

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.5",
    "Referer": "https://www.bbcgoodfood.com/",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

# %%

def getNameTokens(tokens, x=0):
    tokenString = (" ").join(tokens)
    #print(tokenString)
    nameTokens = list(nlp(tokenString).ents)
    i = 0
    while i<len(nameTokens):
        if len(nameTokens[i])>20:
            nameTokens.pop(i)
        else:
            nameTokens[i] = str(nameTokens[i])
            i += 1
    #for i in range(len(nameTokens)):
    #    print(str(type(nameTokens[i]))+":"+str(nameTokens[i]))
    return nameTokens

# %%

def tokenize(text):
    used_stopwords = stopwords.words('english')

    unwanted_punctuation = string.punctuation
    unwanted_punctuation = unwanted_punctuation.replace('\'', '')

    # Simplify text into tokens
    tokens = word_tokenize(text)

    # Special Treatments -----------------------------

    # Names - If two consecutive tokens start with capital letters, this is considered a name
    x = 0
    while x < len(tokens):  # Removes - as they can break up names. While loop used due to changing size of tokens list
        token = tokens[x]
        tokens[x] = tokens[x].replace("-", "")
        if len(tokens[x]) == 0:  # Now an empty string (probably was a stopword with a - attached)
            del tokens[x]
        else:
            x += 1

    tokens += getNameTokens(tokens)  # Adds names to list. Original terms as part of names still remain as seperate entries
    # ------------------------------------------------

    # Makes all tokens lower case
    x = 0
    while x < len(tokens):  # While loop used due to changing size of tokens list
        tokens[x] = tokens[x].lower()
        if len(tokens[x]) > 15:  # Manually removes tokens that are considered unreasonably long
            del tokens[x]
        else:
            x += 1


    # Removes punctuation and stopwords, then simplify to stem
    tokens = [stemmer.stem(c) for c in tokens if (not token in used_stopwords) and (not c in unwanted_punctuation)]  # Punctuation removed after name checks so it can seperate two names properly. Stemming done after
    return tokens

def processFile(file, scrapeForDomain="", maxScrapeDepth=0, useCache=True) -> list:
    """Process a file-like object or file path and return a list of {file, tokens}.

    If scraping is enabled, this will also download child pages (cached under
    `websites/<domain>/`) and include them as separate returned entries.
    """
    # Read HTML from file-like object or path
    if hasattr(file, "read"):
        html = file.read()
        source_name = os.path.relpath(file.name) if hasattr(file, "name") else "inline"
    else:
        # treat 'file' as a path
        with open(file, "r", encoding="utf8") as fh:
            html = fh.read()
        source_name = os.path.relpath(file)

    soup = BeautifulSoup(html, "lxml")
    # When indexing local saved pages we may not have a base URL; derive one
    base_url = None
    if scrapeForDomain:
        base_url = f"https://{scrapeForDomain}"
    # visited set shared across recursion in a single processFile call
    return processSoup(soup, scrapeForDomain, maxScrapeDepth, base_url=base_url, visited=set(), source_name=source_name, useCache=useCache)

def _extract_text_from_jsonld(soup) -> str:
    """Try to pull meaningful text out of any JSON-LD Recipe blocks."""
    text_parts = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string)
        except Exception:
            continue
        if isinstance(data, list):
            data = next((d for d in data if d.get("@type") == "Recipe"), None)
        if data and data.get("@type") == "Recipe":
            for k in ("name", "description", "recipeYield"):
                if data.get(k):
                    text_parts.append(str(data.get(k)))
            for ing in data.get("recipeIngredient", []):
                text_parts.append(str(ing))
            for step in data.get("recipeInstructions", []):
                if isinstance(step, dict):
                    text_parts.append(step.get("text", ""))
                else:
                    text_parts.append(str(step))
            return " ".join(text_parts)
    return ""


def extract_recipe_metadata(soup, url=None):
    """Extract structured recipe metadata from a page.

    Returns a dict with keys: preparation_time (minutes), cooking_time (minutes),
    total_time (minutes), portions_count (int), ingredients (list of {name,quantity,unit}),
    instructions (list of steps), link (string).
    Best-effort: prefers site JSON payloads (props.pageProps), then schema.org JSON-LD,
    then HTML heuristics.
    """
    meta = {
        "preparation_time": None,
        "cooking_time": None,
        "total_time": None,
        "portions_count": None,
        "ingredients": [],
        "instructions": [],
        "link": url or "",
    }

    # 1) Look for site JS payloads: props.pageProps or window.__INITIAL_STATE__ patterns
    scripts = soup.find_all("script")
    for script in scripts:
        txt = script.string
        if not txt:
            continue
        # naive search for props.pageProps JSON
        if "props" in txt and "pageProps" in txt and "ingredients" in txt:
            try:
                # attempt to extract a JSON object starting at first '{'
                start = txt.find('{')
                candidate = txt[start:]
                data = json.loads(candidate)
            except Exception:
                # fallback: try to find a smaller JSON by regex (not perfect)
                try:
                    m = re.search(r"(\{\"props\":.*\})", txt, flags=re.S)
                    if m:
                        data = json.loads(m.group(1))
                    else:
                        data = None
                except Exception:
                    data = None
            if data:
                # drill into common path
                props = data.get("props") or data.get("__INITIAL_STATE__")
                if props:
                    pageProps = props.get("pageProps") or props.get("props")
                    if pageProps:
                        # BBC example: pageProps.page or pageProps.schema
                        # Try multiple known locations
                        candidate = None
                        if isinstance(pageProps, dict):
                            candidate = pageProps
                        else:
                            candidate = None
                        if candidate:
                            # ingredients
                            ings = candidate.get("ingredients") or candidate.get("ingredientsGroups") or candidate.get("ingredientsList")
                            if ings and isinstance(ings, list):
                                for g in ings:
                                    # BBC structure uses nested lists
                                    if isinstance(g, dict) and "ingredients" in g:
                                        for ing in g.get("ingredients", []):
                                            name = ing.get("ingredientText") or ing.get("term", {}).get("display") or ing.get("ingredient") or ing.get("name")
                                            qty = None
                                            unit = None
                                            if ing.get("metricQuantity") is not None:
                                                qty = ing.get("metricQuantity")
                                                unit = ing.get("metricUnit")
                                            elif ing.get("quantityText"):
                                                qty_unit = ing.get("quantityText")
                                                # very small parse: split numeric and unit
                                                m = re.match(r"([0-9/.]+)\s*(.*)", qty_unit)
                                                if m:
                                                    try:
                                                        qty = float(m.group(1))
                                                    except Exception:
                                                        qty = None
                                                    unit = m.group(2).strip() or None
                                            meta["ingredients"].append({"name": name, "quantity": qty, "unit": unit})
                            # method / steps
                            steps = candidate.get("methodSteps") or candidate.get("method") or candidate.get("methodSteps")
                            if steps and isinstance(steps, list):
                                for s in steps:
                                    # BBC uses list of {type:step, content:[{type:html,data:{value:...}}]}
                                    if isinstance(s, dict):
                                        if s.get("content"):
                                            for c in s.get("content"):
                                                if isinstance(c, dict) and c.get("data") and c["data"].get("value"):
                                                    # strip html
                                                    txt = BeautifulSoup(c["data"]["value"], "lxml").get_text(strip=True)
                                                    if txt:
                                                        meta["instructions"].append(txt)
                                        elif s.get("text"):
                                            meta["instructions"].append(s.get("text"))
                                
                            # times and servings
                            cook_p = candidate.get("cookAndPrepTime") or candidate.get("recipe") or {}
                            if isinstance(cook_p, dict):
                                if cook_p.get("preparationMin") is not None:
                                    meta["preparation_time"] = int(cook_p.get("preparationMin"))
                                elif cook_p.get("prep_time") is not None:
                                    meta["preparation_time"] = int(cook_p.get("prep_time"))
                                if cook_p.get("cookingMin") is not None:
                                    meta["cooking_time"] = int(cook_p.get("cookingMin"))
                                if cook_p.get("total") is not None:
                                    meta["total_time"] = int(cook_p.get("total"))
                            if candidate.get("servings"):
                                try:
                                    meta["portions_count"] = int(candidate.get("servings"))
                                except Exception:
                                    pass

    # 2) schema.org JSON-LD
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string)
        except Exception:
            continue
        if isinstance(data, list):
            data = next((d for d in data if d.get("@type") == "Recipe"), None)
        if data and data.get("@type") == "Recipe":
            # times: PT20M etc. convert to minutes
            def parse_duration(pt):
                if not pt:
                    return None
                m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?", pt)
                if not m:
                    return None
                hours = int(m.group(1)) if m.group(1) else 0
                mins = int(m.group(2)) if m.group(2) else 0
                return hours * 60 + mins

            prep = parse_duration(data.get("prepTime") or data.get("preparationTime"))
            cook = parse_duration(data.get("cookTime") or data.get("cookingTime"))
            total = parse_duration(data.get("totalTime"))
            if prep:
                meta["preparation_time"] = prep
            if cook:
                meta["cooking_time"] = cook
            if total:
                meta["total_time"] = total

            # servings
            if data.get("recipeYield"):
                try:
                    meta["portions_count"] = int(re.sub(r"\D", "", str(data.get("recipeYield"))))
                except Exception:
                    pass

            # ingredients
            for ing in data.get("recipeIngredient", []):
                # naive parse: "200g strawberries hulled and chopped" -> qty 200 unit g name strawberries + note
                qty = None
                unit = None
                name = ing
                m = re.match(r"^\s*([0-9/.]+)\s*([a-zA-Z]+)\s+(.*)", ing)
                if m:
                    try:
                        qty = float(m.group(1))
                    except Exception:
                        qty = None
                    unit = m.group(2)
                    name = m.group(3)
                meta["ingredients"].append({"name": name, "quantity": qty, "unit": unit})

            # instructions
            instr = data.get("recipeInstructions") or []
            for step in instr:
                if isinstance(step, dict):
                    if step.get("text"):
                        meta["instructions"].append(step.get("text"))
                else:
                    meta["instructions"].append(str(step))

            # link
            if data.get("url"):
                meta["link"] = data.get("url")

            break

    # 3) HTML fallbacks: look for ingredient lists and method lists
    if not meta["ingredients"]:
        # common selectors
        for sel in [".ingredients", ".recipe-ingredients", "#ingredients", ".ingredients-list"]:
            block = soup.select_one(sel)
            if block:
                for li in block.find_all("li"):
                    txt = li.get_text(strip=True)
                    if txt:
                        meta["ingredients"].append({"name": txt, "quantity": None, "unit": None})
                if meta["ingredients"]:
                    break

    if not meta["instructions"]:
        for sel in [".method", ".method-steps", ".directions", ".instructions", "#method"]:
            block = soup.select_one(sel)
            if block:
                for li in block.find_all(["li", "p"]):
                    txt = li.get_text(strip=True)
                    if txt:
                        meta["instructions"].append(txt)
                if meta["instructions"]:
                    break

    # compute total if missing
    try:
        if meta["total_time"] is None:
            pt = meta.get("preparation_time") or 0
            ct = meta.get("cooking_time") or 0
            if pt or ct:
                meta["total_time"] = int(pt + ct)
    except Exception:
        pass

    return meta


def processSoup(soup, scrapeForDomain="", maxScrapeDepth=0, base_url=None, visited=None, source_name=None, useCache=True) -> list:
    # Attempts to get the most relevant starting point to search through
    if soup.find("main"):
        main = soup.find("main")
    elif soup.find("div",{"id":"page"}):
        main = soup.find("div",{"id":"page"})
    else:
        main = soup.find("body") # Defaults to body

    # Cleanup
    divs = soup.find_all('div')
    for div in divs:
        if "style" in div:
            if "display:none" in div["style"]: # Removes hidden divs
                div.decompose()
    for nav in main.find_all("nav"): # Removes all navs as these are usually menus
        nav.decompose()
    # Adds all the relevant text to a string called text
    relevant = main.find_all({re.compile('^h[1-6]$'),"p","li"})
    text = ""
    for element in relevant:
        text += (" ").join(element.find_all(string=True)) # Adds new line so the last word of this element and the first word of the next don't join

    a_elems = main.find_all("a")
    # Prepare visited set to avoid repeated fetches
    if visited is None:
        visited = set()

    results = []

    # Main page tokens (use provided source_name or a generic placeholder)
    page_key = source_name or (f"{urlparse(base_url).netloc}/root" if base_url else "inline")
    main_tokens = tokenize(text)
    results.append({"file": page_key, "tokens": main_tokens})
    # Extract recipe metadata for the main page and attach to results entries as 'meta'
    try:
        main_meta = extract_recipe_metadata(soup, url=(base_url or ""))
        results[-1]["meta"] = main_meta
    except Exception as e:
        results[-1]["meta"] = {}

    def sanitize_filename(url: str) -> str:
        p = urlparse(url)
        path_part = p.path.strip('/').replace('/', '_')
        if not path_part:
            path_part = 'index'
        h = hashlib.sha1(url.encode('utf-8')).hexdigest()[:10]
        return f"{path_part}_{h}.html"

    for element in a_elems:
        # Add link text for non-menu links to main tokens (heuristic)
        if element.parent.name != "li":
            # append small amount of link text to main tokens to preserve context
            link_text = element.get_text(strip=True)
            if link_text:
                # we don't retokenize here; just append raw text to be safe
                main_tokens += tokenize(link_text)

        # Scrape linked pages when requested
        if scrapeForDomain and maxScrapeDepth > 0:
            href = element.get("href")
            if not href:
                continue

            # Build absolute URL
            if href.startswith("//"):
                href = "https:" + href
            if href.startswith("/") and base_url:
                full_link = urljoin(base_url, href)
            elif href.startswith("http"):
                full_link = href
            else:
                # Relative link without base; try joining to provided base_url
                full_link = urljoin(base_url or "", href)

            # Filter to domain
            try:
                netloc = urlparse(full_link).netloc
            except Exception:
                netloc = ""
            if scrapeForDomain not in full_link and scrapeForDomain not in netloc:
                continue

            # Avoid refetching same URL
            norm = full_link.split('#')[0].rstrip('/')
            if norm in visited:
                continue
            visited.add(norm)

            try:
                print(f"Scraping {full_link} (depth {maxScrapeDepth})")
                # Determine cache path
                cache_dir = os.path.join("websites", netloc)
                os.makedirs(cache_dir, exist_ok=True)
                filename = sanitize_filename(full_link)
                cache_path = os.path.join(cache_dir, filename)

                if useCache and os.path.exists(cache_path):
                    with open(cache_path, "r", encoding="utf-8") as fh:
                        child_html = fh.read()
                    child_soup = BeautifulSoup(child_html, "lxml")
                else:
                    # Polite delay between requests
                    time.sleep(random.uniform(0.5, 1.3))
                    resp = SESSION.get(full_link, timeout=10)
                    resp.raise_for_status()
                    child_html = resp.text
                    child_soup = BeautifulSoup(child_html, "lxml")
                    try:
                        with open(cache_path, "w", encoding="utf-8") as fh:
                            fh.write(child_html)
                    except Exception as e:
                        print(f"Warning: failed to write cache {cache_path}: {e}")

                # Prefer structured JSON-LD content when available; if present, produce single token entry
                jsonld_text = _extract_text_from_jsonld(child_soup)
                if jsonld_text:
                    child_tokens = tokenize(jsonld_text)
                    child_key = os.path.relpath(cache_path)
                    child_meta = {}
                    try:
                        child_meta = extract_recipe_metadata(child_soup, url=full_link)
                    except Exception:
                        child_meta = {}
                    results.append({"file": child_key, "tokens": child_tokens, "meta": child_meta})
                else:
                    # Recurse into the child page, passing along visited set
                    child_base = f"{urlparse(full_link).scheme}://{urlparse(full_link).netloc}"
                    child_results = processSoup(child_soup, scrapeForDomain, maxScrapeDepth-1, base_url=child_base, visited=visited, source_name=os.path.relpath(cache_path), useCache=useCache)
                    results.extend(child_results)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                print(f"Error scraping {full_link}: {e} on line {exc_tb.tb_lineno}")

    return results


# %%

def generateIndexes(useCache=True):
    global postings
    global docID
    global vocabID
    # Processes every file in the wiki folder

    # Adds terms to the index
    folder_name = "websites/startingpages"
    #folder_name = "../ueasmall"
    print("Processing "+str(len(os.listdir(folder_name)))+" files...")
    for file in os.listdir(folder_name):
        file_path = os.path.join(folder_name, file)
        with open(file_path, "r", encoding="utf8") as f:
            entries = processFile(f, scrapeForDomain="https://www.hellofresh.co.uk", maxScrapeDepth=2, useCache=useCache)

        # entries is a list of {file: str, tokens: list}
        for entry in entries:
            file_key = entry.get("file")
            tokens = entry.get("tokens", [])
            entry_meta = entry.get("meta", {})

            # Adds file to docID
            if file_key not in docID:
                docID[file_key] = len(docID)
            d = docID[file_key]

            # Map metadata to integer docID in docInfo
            try:
                if entry_meta:
                    # ensure docInfo keys are strings for JSON-friendly storage
                    docInfo[str(d)] = entry_meta
                elif str(d) not in docInfo:
                    docInfo[str(d)] = {}
            except Exception:
                pass

            for term in tokens:  # Loops through and adds occurrence of term into index
                # Gets vocabID
                if term not in vocabID:
                    vocabID[term] = len(vocabID)
                t = vocabID[term]

                # Adds term to postings
                if t not in postings:
                    postings[t] = {d: {"frequency": 0}}  # Makes new entry in postings for term for the page with frequency set to 0 to start with
                if d not in postings[t]:
                    postings[t][d] = {"frequency": 0}  # Makes new entry for the term with frequency set to 0 to start with
                page = postings[t][d]
                page["frequency"] += 1

    # Saves postings
    print("Saving data")
    with open("indexes/postings.json", "w", encoding='utf-8') as file:
        json.dump(postings, file, indent=4)
    # Saves docIDs
    with open("indexes/docID.json", "w", encoding='utf-8') as file:
        json.dump(docID, file, indent=4)
    # Saves vocabIDs
    with open("indexes/vocabID.json", "w", encoding='utf-8') as file:
        json.dump(vocabID, file, indent=4)
    # Saves docInfo
    with open("indexes/docInfo.json", "w", encoding='utf-8') as file:
        json.dump(docInfo, file, indent=4)
    print("Saved")

# %%

def tf_idf(term_freq, doc_freq, N):
    tfidf = log(1+term_freq) * log(N/doc_freq)
    if term_freq != 0:
        return tfidf
    else:
        return 0

# %%

def queryItems(q):
    global docID
    global vocabID
    global postings
    results = {}
    if type(q) != str:  # Tokenises terms if need be
        terms = q  # Terms may be passed in as list from other instances of this function
    else:
        terms = tokenize(q)
        #print("Start of query")
    print(terms)

    # Searches for terms
    if len(terms) > 1:  # Multi-term query found, will be fed into the recursive process
        docsQ1 = queryItems([terms[0]])  # Gets results of docs with the first term using 1 more recursion
        docsQ2 = queryItems(terms[1:]) # Gets results of docs for the rest of the terms using multiple recursions

        # If a term isn't found, it is ignored # TODO: remove this bit. seems useless
        if terms[0] not in vocabID:
            print(str(terms[0])+" not found")

        for doc in docsQ2: # Combines results
            if doc in docsQ1:
                #print("Doc:\n" +doc+ terms[0] + str(docsQ1[doc]) + "\nAND\n" + str(terms[2:]) + str(docsQ2[doc]))
                docsQ1[doc]["score"] += docsQ2[doc]["score"]
            else:
                docsQ1[doc] = docsQ2[doc]
        return docsQ1

    # Single word query found. Will be formatted to string
    q = ''.join(terms)

    # Base case
    q = q.lower()
    if q in vocabID:  # Known term
        t = str(vocabID[q])
    else:
        return {}
    if t in postings:  # Gets occurrences into results
        for d in postings[t]:
            results[d] = postings[t][d]
            #print("Doc: "+d+str(results[d]))
            if "score" not in results[d]:
                results[d]["score"] = 0
            #print("tf-idf"+str(d)+": "+str(tf_idf(results[d]["frequency"],len(postings[t]),len(docID))))
            results[d]["score"] += tf_idf(results[d]["frequency"],len(postings[t]),len(docID)) # Adds tf-idf to relevancy score
            #print("Results:"+str(results[d]))
    return results

# %%

def sortByFreq(results):  # Basic insertion sort
    rList = []
    for result in results:
        rList.append({result: results[result]})
    if len(results) == 0:
        return []
    sorted = [rList[0]]
    for x in range(1, len(rList)):
        pos = len(sorted)
        for y in range(len(sorted)):
            if list(rList[x].values())[0]["frequency"] > list(sorted[y].values())[0]["frequency"]:
                pos = y
                break
        sorted.insert(pos, rList[x])
    return sorted

def sortByScore(results):  # Basic insertion sort
    rList = []
    print("Sorting")
    for result in results:
        rList.append({result: results[result]})
    if len(results) == 0:
        return []
    sorted = [rList[0]]
    for x in range(1, len(rList)):
        pos = len(sorted)
        for y in range(len(sorted)):
            if list(rList[x].values())[0]["score"] > list(sorted[y].values())[0]["score"]:
                pos = y
                break
            elif list(rList[x].values())[0]["score"] == list(sorted[y].values())[0]["score"]:
                if list(rList[x].values())[0]["frequency"] > list(sorted[y].values())[0]["frequency"]: # Frequency used if scores are the same
                    pos = y
                    break
        sorted.insert(pos, rList[x])
    return sorted

# %%

def query(q):
    global docID
    global postings
    # Prepares query for final results
    results = sortByScore(queryItems(q))
    formattedResults = []
    docIDInv = {d: i for i, d in docID.items()}
    for item in results:  # Swaps out docID for doc name
        key = list(item.keys())[0]
        formatted = {docIDInv[int(key)]: item}
        formatted = {str(docIDInv[int(key)]):str(formatted[docIDInv[int(key)]][key]["score"])}
        formattedResults.append(formatted)
    if len(formattedResults) > 10:
        formattedResults = formattedResults[:10]
    return formattedResults

# %%

docID = {}
postings = {}
vocabID = {}
docInfo = {}  # New dictionary to store additional document information (Preparation time, cooking time, total time, portions count, ingredients, intructions, link)

def loadData():
    global docID,postings,vocabID,docInfo
    with open("indexes/postings.json", "r", encoding='utf-8') as file:
        postings = json.load(file)
    with open("indexes/docID.json", "r", encoding='utf-8') as file:
        docID = json.load(file)
    with open("indexes/vocabID.json", "r", encoding='utf-8') as file:
        vocabID = json.load(file)
    with open("indexes/docInfo.json", "r", encoding='utf-8') as file:
        docInfo = json.load(file)

# %%

def clearData():
    print("Deleting data")
    with open("indexes/postings.json", "w", encoding='utf-8') as file:
        json.dump({}, file, indent=4)
    # Saves docIDs
    with open("indexes/docID.json", "w", encoding='utf-8') as file:
        json.dump({}, file, indent=4)
    # Saves vocabIDs
    with open("indexes/vocabID.json", "w", encoding='utf-8') as file:
        json.dump({}, file, indent=4)
    # Saves docInfo
    with open("indexes/docInfo.json", "w", encoding='utf-8') as file:
        json.dump({}, file, indent=4)
    print("Deleted")

# %%

def main_cli():
    """Interactive console entrypoint. Guarded so importing this module
    from other code (like an API server) doesn't start the CLI loop.
    """
    command = ""
    loadData()
    while command != "exit":
        command = input("command: ")
        t0 = time.perf_counter()
        if command == "process":
            clearData()
            generateIndexes()
            loadData()
        elif command == "query":
            command = input("query: ")
            while command != "<":
                t0 = time.perf_counter()
                loadData()  # Resets document scores
                results = query(command)
                print("Results: " + str(results))

                t1 = time.perf_counter()
                print("Time taken: " + str(round(t1 - t0, 100)) + " seconds")
                command = input("query: ")
        elif command == "clear":
            clearData()
        elif command == "help":
            print("\033[1mProgram functions:\033[0m")
            print("exit - exits program")
            print("process - indexes and processes files")
            print("query - enters query mode (exit by entering \"<\")")
            print("clear - clears all stored data")
            print("help - well I think you already know what this does")
            print("")

        t1 = time.perf_counter()
        print("Time taken: " + str(round(t1 - t0, 100)) + " seconds")


if __name__ == "__main__":
    main_cli()
