import pytest

from competitor.recommendation_placements import (
    RecommendationPlacement,
    RecommendationPlacementDetector,
    detect_recommendation_placements,
)


def product_cards(count=3, card_class="product-card"):
    return "".join(
        """
        <article class="{card_class}" data-product-id="sku-{number}">
          <a href="/products/item-{number}">
            <img src="/images/{number}.jpg" alt="Item {number}">
            <h3>Item {number}</h3>
          </a>
          <span class="price">${price}.00</span>
        </article>
        """.format(card_class=card_class, number=number, price=number * 10)
        for number in range(1, count + 1)
    )


@pytest.mark.parametrize(
    "title, expected",
    [
        ("Popular", "Popular"),
        ("Our Most Popular Products", "Popular"),
        ("Trending Now", "Trending"),
        ("Recently Viewed", "Recently Viewed"),
        ("You May Also Like", "You May Also Like"),
        ("Similar Products", "Similar Products"),
        ("Frequently Bought Together", "Frequently Bought Together"),
        ("Customers Also Bought", "Customers Also Bought"),
        ("Recommended For You", "Personalized"),
        ("Featured products", "Unknown"),
    ],
)
def test_classifies_supported_recommendation_labels(title, expected):
    assert RecommendationPlacementDetector.classify_title(title) == expected


def test_detects_multiple_labeled_placements_in_dom_order():
    html = """
    <html><body>
      <main>
        <article class="pdp"><h1>Trail Shoe</h1><button>Add to cart</button></article>
        <section id="similar-products" class="recommendation-carousel">
          <h2>Similar Products</h2>
          <div class="slides">{similar}</div>
        </section>
        <section data-component="frequently-bought-together">
          <h2>Frequently Bought Together</h2>
          <div class="bundle">{bundle}</div>
        </section>
      </main>
    </body></html>
    """.format(similar=product_cards(4), bundle=product_cards(3, "sku-tile"))

    placements = detect_recommendation_placements(
        html, "https://shop.example/products/trail-shoe", "product"
    )

    assert [placement.detected_type for placement in placements] == [
        "Similar Products",
        "Frequently Bought Together",
    ]
    assert [placement.placement_position for placement in placements] == [1, 2]
    assert [placement.product_count for placement in placements] == [4, 3]
    assert all(placement.page_type == "product" for placement in placements)
    assert all(placement.page_url.endswith("/trail-shoe") for placement in placements)
    assert all(0.0 <= placement.confidence <= 1.0 for placement in placements)
    assert any("heading" in item for item in placements[0].supporting_dom_evidence)
    assert any("carousel" in item for item in placements[0].supporting_dom_evidence)


def test_uses_accessible_and_component_labels_without_visible_heading():
    html = """
    <div role="region" aria-label="Recently Viewed" class="swiper">
      <div class="swiper-wrapper">
        <div class="swiper-slide"><a href="/products/a"><img src="a.jpg"></a><b>$10</b></div>
        <div class="swiper-slide"><a href="/products/b"><img src="b.jpg"></a><b>$20</b></div>
      </div>
    </div>
    """

    placements = RecommendationPlacementDetector().detect(
        html, "https://shop.example/cart", "cart"
    )

    assert len(placements) == 1
    assert placements[0].placement_title == "Recently Viewed"
    assert placements[0].detected_type == "Recently Viewed"
    assert placements[0].product_count == 2
    assert any("slider" in evidence for evidence in placements[0].supporting_dom_evidence)


def test_reports_unlabeled_repeated_product_section_as_unknown():
    html = """
    <section class="discovery-shelf">
      <h2>Explore more</h2>
      <div class="tiles">
        <div class="tile"><a href="/products/a"><img src="a.jpg">A</a><span>$10</span></div>
        <div class="tile"><a href="/products/b"><img src="b.jpg">B</a><span>$20</span></div>
        <div class="tile"><a href="/products/c"><img src="c.jpg">C</a><span>$30</span></div>
      </div>
    </section>
    """

    placements = detect_recommendation_placements(
        html, "https://shop.example/products/a", "product"
    )

    assert len(placements) == 1
    assert placements[0].placement_title == "Explore more"
    assert placements[0].detected_type == "Unknown"
    assert placements[0].product_count == 3
    assert any(
        "repeated product structure" in evidence
        for evidence in placements[0].supporting_dom_evidence
    )


def test_ignores_generic_catalog_and_heading_without_products():
    html = """
    <h2>You May Also Like</h2>
    <p>Sign up for our newsletter.</p>
    <section class="catalog">
      <h2>Products</h2>
      <div class="grid">{cards}</div>
    </section>
    <footer>
      <section class="popular-links">
        <h2>Popular</h2><a href="/help">Help</a><a href="/contact">Contact</a>
      </section>
    </footer>
    """.format(cards=product_cards(6))

    assert detect_recommendation_placements(
        html, "https://shop.example/collections/shoes", "category"
    ) == []


def test_placement_serialization_round_trip():
    placement = RecommendationPlacement(
        page_url="https://shop.example/p/1",
        page_type="product",
        placement_title="Just for you",
        placement_position=2,
        product_count=5,
        detected_type="Personalized",
        confidence=0.94,
        supporting_dom_evidence=["recommendation heading"],
    )

    assert RecommendationPlacement.from_dict(placement.to_dict()) == placement
