"""Pauli quicklook display for the products found by the search step."""

import urllib.request
from io import BytesIO

import matplotlib.pyplot as plt
from PIL import Image

QUICKLOOK_KEY = "enclosure_ql_pauli_png"


def _fetch_bytes(url, token):
    """Download a URL, authenticated (assets need it) with anonymous fallback."""
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token.get()}"})
    try:
        with urllib.request.urlopen(request) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        if exc.code not in (401, 403):
            raise
        with urllib.request.urlopen(url) as response:  # some assets are public
            return response.read()


def show_quicklooks(config, roi_entry, token, limit=None, show=False):
    """Download Pauli quicklooks of a ROI's products into <roi>/quicklooks/.

    limit=None downloads all products of the ROI entry.
    """
    products = roi_entry.get("products", [])
    if not products:
        print("No products, skipping quicklooks")
        return []

    out_dir = config.roi_dir(roi_entry["slug"]) / "quicklooks"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = roi_entry["name"]
    selected = products if limit is None else products[:limit]
    saved = []

    for i, product in enumerate(selected):
        asset_url = product["assets"].get(QUICKLOOK_KEY)
        if not asset_url:
            print(f"  -> no {QUICKLOOK_KEY} asset for {product['id']}")
            continue

        image = Image.open(BytesIO(_fetch_bytes(asset_url, token)))

        date = (product.get("datetime") or "")[:10]
        out_path = out_dir / f"{product['id'][:40]}_{date}.png"
        if not out_path.exists():
            image.save(out_path)
        saved.append(out_path)
        print(f"  saved {out_path.name}")

        if show:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(image)
            ax.set_title(f"{name}\n{product['datetime']} (Pauli RGB)")
            ax.axis("off")
            fig.tight_layout()
            plt.show()
            plt.close(fig)

    print(f"Products in catalogue: {len(products)} | first: {products[0]['datetime']} | last: {products[-1]['datetime']}")
    return saved


def quicklooks_for_stack(config, roi_entry, stack_info, token, show=False):
    """Download the quicklook of each product of an extracted stack into its folder.

    stack_info: the parsed stack_info.json of the stack folder.
    Failures (missing asset, bad token) are reported but do not stop the pipeline.
    """
    by_id = {p["id"]: p for p in roi_entry.get("products", [])}
    out_dir = stack_info["_dir"] / "quicklooks"
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for product in stack_info["products"]:
        catalogue_product = by_id.get(product["id"])
        asset_url = catalogue_product and catalogue_product["assets"].get(QUICKLOOK_KEY)
        if not asset_url:
            print(f"  -> no quicklook URL for {product['id']}")
            continue
        date = (product.get("datetime") or "")[:10]
        out_path = out_dir / f"{product['id'][:40]}_{date}.png"
        if out_path.exists():
            saved.append(out_path)
            continue
        try:
            image = Image.open(BytesIO(_fetch_bytes(asset_url, token)))
            image.save(out_path)
        except Exception as exc:
            print(f"  -> quicklook failed for {product['id']}: {exc}")
            continue
        saved.append(out_path)
        print(f"  saved {out_path.name}")
    return saved
