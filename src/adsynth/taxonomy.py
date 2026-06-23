"""The 23 ad-asset categories and fuzzy mapping from free-form asset names.

The concepts.json asset names are free-form ("Background Animation", "Tagline",
"Interactive Element", …). We normalize them to a stable taxonomy so the layout
engine can reason about roles (where does a Logo go vs. a Background).
"""
from __future__ import annotations

# Canonical categories (from data/categories.txt).
CATEGORIES: list[str] = [
    "Background Image",
    "Logo",
    "Call-To-Action (CTA) Button",
    "Icon",
    "Product Image",
    "Text Elements",
    "Infographic",
    "Banner",
    "Illustration",
    "Photograph",
    "Mascot",
    "Testimonial Quotes",
    "Social Proof",
    "Seal or Badge",
    "Graphs and Charts",
    "Decorative Elements",
    "Interactive Elements",
    "Animation Frames",
    "Coupon or Offer Code",
    "Legal Disclaimers or Terms",
    "Contact Information",
    "Map or Location Image",
    "QR Code",
]

# keyword -> canonical category. First match wins (longer/more specific first).
_KEYWORD_MAP: list[tuple[tuple[str, ...], str]] = [
    (("cta", "call-to-action", "call to action", "play now", "button"), "Call-To-Action (CTA) Button"),
    (("logo", "brand mark", "wordmark"), "Logo"),
    (("background", "backdrop", "scene", "environment", "foreground"), "Background Image"),
    (("tagline", "headline", "copy", "text", "title", "caption", "slogan", "message"), "Text Elements"),
    (("product",), "Product Image"),
    (("mascot", "character"), "Mascot"),
    (("testimonial", "review", "quote"), "Testimonial Quotes"),
    (("social proof", "rating", "stars"), "Social Proof"),
    (("badge", "seal", "award"), "Seal or Badge"),
    (("chart", "graph", "infographic", "stat"), "Infographic"),
    (("coupon", "offer", "discount", "promo code"), "Coupon or Offer Code"),
    (("legal", "disclaimer", "terms", "t&c"), "Legal Disclaimers or Terms"),
    (("contact", "phone", "email", "address"), "Contact Information"),
    (("map", "location"), "Map or Location Image"),
    (("qr",), "QR Code"),
    (("banner",), "Banner"),
    (("illustration", "drawing", "graphic"), "Illustration"),
    (("photo", "photograph", "image of", "picture"), "Photograph"),
    (("icon", "symbol", "indicator", "arrow", "swipe", "tap"), "Icon"),
    (("animation", "video", "clip", "frame", "transition", "effect", "sound"), "Animation Frames"),
    (("interactive", "game", "slider", "wheel", "drag", "quiz", "toolbox"), "Interactive Elements"),
    (("decorative", "ornament", "pattern", "confetti", "sparkle"), "Decorative Elements"),
]


def normalize_category(raw_name: str) -> str:
    """Map a free-form asset name to one of the 23 canonical categories."""
    name = (raw_name or "").lower()
    for keywords, category in _KEYWORD_MAP:
        if any(k in name for k in keywords):
            return category
    # exact-ish fallback against canonical names
    for cat in CATEGORIES:
        if cat.lower() in name or name in cat.lower():
            return cat
    return "Decorative Elements"
