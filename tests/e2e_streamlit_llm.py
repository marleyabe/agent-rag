from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from playwright.sync_api import Page, expect, sync_playwright


APP_URL = os.getenv("E2E_APP_URL", "http://localhost:8504")
PDF_PATH = Path(os.getenv("E2E_PDF_PATH", "cartilha_ppsi.pdf")).resolve()
SCREENSHOT_PATH = Path(os.getenv("E2E_SCREENSHOT_PATH", "/tmp/streamlit-e2e-llm-final.png"))


def assert_no_visible_app_error(page: Page) -> None:
    errors = page.locator("text=/Nao foi possivel|Traceback|Exception|ModuleNotFoundError/i")
    if errors.count() > 0 and errors.first.is_visible():
        raise AssertionError(errors.first.inner_text())


def assert_real_llm_answer(page: Page) -> None:
    page_text = page.locator("body").inner_text()
    if "conforme os documentos enviados" in page_text:
        raise AssertionError("The answer came from the local FakeLLM fallback.")
    if "Nao encontrei essa informacao nos documentos enviados" in page_text:
        raise AssertionError("The LLM flow did not produce an evidenced answer.")
    if not re.search(r"maturidade|resili|privacidade|seguran", page_text, re.I):
        raise AssertionError("The answer did not contain expected PPSI concepts.")


def main() -> int:
    if not PDF_PATH.exists():
        raise FileNotFoundError(PDF_PATH)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=os.getenv("E2E_CHROMIUM_PATH", "/usr/bin/chromium-browser"),
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.set_default_timeout(180_000)

        page.goto(APP_URL, wait_until="domcontentloaded")
        expect(page.get_by_text("RAG Notebook MVP")).to_be_visible()
        assert_no_visible_app_error(page)

        page.locator("input[type='file']").set_input_files(str(PDF_PATH))
        page.get_by_role("button", name="Indexar documento").click()
        expect(page.get_by_text(re.compile(r"Documento indexado:.*cartilha_ppsi", re.I))).to_be_visible()
        assert_no_visible_app_error(page)

        chat_input = page.locator("textarea[placeholder='Pergunte sobre os documentos']")
        expect(chat_input).to_be_visible()
        chat_input.fill("Qual e o objetivo do PPSI? Responda de forma objetiva.")
        chat_input.press("Enter")

        expect(page.get_by_text("Qual e o objetivo do PPSI? Responda de forma objetiva.")).to_be_visible()
        expect(page.locator("a[href*='doc=']").first).to_be_visible(timeout=180_000)
        assert_no_visible_app_error(page)
        assert_real_llm_answer(page)

        first_citation = page.locator("a[href*='doc=']").first
        href = first_citation.get_attribute("href")
        if not href:
            raise AssertionError("Citation link did not expose an href")
        first_citation.click()
        try:
            page.wait_for_url(re.compile(r".*\?doc=.*chunk=.*"), timeout=30_000)
        except Exception:
            page.goto(f"{APP_URL}/{href}" if href.startswith("?") else href)
        expect(page.get_by_text("Visualizacao de citacao")).to_be_visible(timeout=60_000)
        expect(page.get_by_text("Arquivo: cartilha_ppsi.pdf")).to_be_visible(timeout=60_000)
        assert_no_visible_app_error(page)

        page.screenshot(path=str(SCREENSHOT_PATH), full_page=True)
        browser.close()

    print(f"LLM E2E passed. Screenshot: {SCREENSHOT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
