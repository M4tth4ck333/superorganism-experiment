from __future__ import annotations

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QScrollArea,
    QFrame,
    QGridLayout,
    QSizePolicy,
    QToolButton,
)

from ui.common.icons import icon, icon_size
from ui.constants import WHITEPAPER_URL


class LandingNavBar(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("landingNavBar")
        self.setFixedHeight(88)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(48, 24, 48, 24)
        layout.setSpacing(16)

        self.brand_lbl = QLabel("SUPERORGANISM")
        self.brand_lbl.setObjectName("landingBrand")

        self.sign_in_btn = QPushButton("Sign In")
        self.sign_in_btn.setProperty("variant", "landing-nav-link")
        self.sign_in_btn.setFixedHeight(40)
        self.sign_in_btn.clicked.connect(lambda: print("Sign In clicked"))

        self.join_btn = QPushButton("Join Mesh")
        self.join_btn.setObjectName("navJoinButton")
        self.join_btn.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.join_btn.setFixedHeight(44)
        self.join_btn.clicked.connect(lambda: print("Join Mesh clicked"))

        layout.addWidget(self.brand_lbl, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addStretch()
        layout.addWidget(self.sign_in_btn, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.join_btn, 0, Qt.AlignmentFlag.AlignVCenter)


class HeroSection(QFrame):
    def __init__(self, image_path: str | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("heroSection")

        self._bg_label = QLabel(self)
        self._bg_label.setObjectName("heroBackground")

        self._overlay = QWidget(self)
        self._overlay.setObjectName("heroOverlay")
        self._overlay.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self._content = QWidget(self)
        self._content.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        content_layout = QVBoxLayout(self._content)
        content_layout.setContentsMargins(64, 140, 64, 80)
        content_layout.setSpacing(22)
        content_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.text_block = QWidget()
        self.text_block.setObjectName("heroTextBlock")
        self.text_block.setMaximumWidth(980)
        self.text_block.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

        text_block_layout = QVBoxLayout(self.text_block)
        text_block_layout.setContentsMargins(0, 0, 0, 0)
        text_block_layout.setSpacing(18)

        self.kicker_lbl = QLabel("Decentralized Protocol Alpha")
        self.kicker_lbl.setObjectName("heroKicker")
        self.kicker_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.title_lbl = QLabel(
            """
            <p align="center" style="line-height: 0.85; margin: 0;">
                The World\'s First<br>
                <span style="color:#b6a0ff;">Autonomous<br>Seedbox.</span>
            </p>
            """
        )
        self.title_lbl.setObjectName("heroTitle")
        self.title_lbl.setTextFormat(Qt.TextFormat.RichText)
        self.title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_lbl.setWordWrap(False)
        self.title_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        self.subtitle_lbl = QLabel(
            "Decentralized, democratic, and indestructible. "
            "Witness the evolution of digital sovereignty."
        )
        self.subtitle_lbl.setObjectName("heroSubtitle")
        self.subtitle_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle_lbl.setWordWrap(True)
        self.subtitle_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        text_block_layout.addWidget(self.kicker_lbl)
        text_block_layout.addWidget(self.title_lbl)
        text_block_layout.addWidget(self.subtitle_lbl)

        button_row = QHBoxLayout()
        button_row.setSpacing(18)
        button_row.setAlignment(Qt.AlignmentFlag.AlignCenter)

        icon_gap = "  "

        self.join_btn = QPushButton(f"Join the Mesh{icon_gap}")
        self.join_btn.setProperty("variant", "landing-primary-hero")
        self.join_btn.setIcon(icon("bolt"))
        self.join_btn.setIconSize(icon_size(18))
        self.join_btn.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self.join_btn.clicked.connect(lambda: print("Hero Join the Mesh clicked"))

        self.whitepaper_btn = QPushButton(f"View Whitepaper{icon_gap}")
        self.whitepaper_btn.setProperty("variant", "landing-secondary-hero")
        self.whitepaper_btn.setIcon(icon("arrow-up-right"))
        self.whitepaper_btn.setIconSize(icon_size(18))
        self.whitepaper_btn.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self.whitepaper_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(WHITEPAPER_URL)))

        button_row.addWidget(self.join_btn)
        button_row.addWidget(self.whitepaper_btn)

        content_layout.addStretch()
        content_layout.addWidget(self.text_block, 0, Qt.AlignmentFlag.AlignHCenter)
        content_layout.addSpacing(8)
        content_layout.addLayout(button_row)
        content_layout.addStretch()

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self._content)

        self._bg_label.lower()
        self._overlay.raise_()
        self._content.raise_()

        self._image_path = image_path
        self._update_background()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._bg_label.setGeometry(self.rect())
        self._overlay.setGeometry(self.rect())
        self._content.setGeometry(self.rect())
        self._update_background()

    def _update_background(self) -> None:
        if not self._image_path:
            return

        pixmap = QPixmap(self._image_path)
        if pixmap.isNull():
            return

        scaled = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._bg_label.setPixmap(scaled)


class SectionHeader(QWidget):
    def __init__(self, title: str, accent: bool = False, label: str | None = None, parent: QWidget | None = None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        if label:
            kicker = QLabel(label)
            kicker.setObjectName("sectionKicker")
            layout.addWidget(kicker)

        title_lbl = QLabel(title)
        title_lbl.setObjectName("sectionTitle")
        layout.addWidget(title_lbl)

        if accent:
            line = QFrame()
            line.setObjectName("sectionAccentLine")
            line.setFixedWidth(96)
            line.setFixedHeight(4)
            layout.addWidget(line)


class PricingCard(QFrame):
    def __init__(
        self,
        plan_name: str,
        price: str,
        suffix: str,
        features: list[str],
        button_text: str,
        highlighted: bool = False,
        badge_text: str | None = None,
        note_text: str | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setProperty("variant", "pricing-card")
        self.setProperty("highlighted", highlighted)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(18)

        if badge_text:
            badge_row = QHBoxLayout()
            badge_row.setContentsMargins(0, 0, 0, 0)
            badge_row.addStretch()

            badge = QLabel(badge_text)
            badge.setObjectName("pricingBadge")
            badge_row.addWidget(badge)

            layout.addLayout(badge_row)

        self.plan_lbl = QLabel(plan_name)
        self.plan_lbl.setObjectName("pricingPlan")
        self.plan_lbl.setProperty("highlighted", highlighted)

        price_row = QHBoxLayout()
        price_row.setContentsMargins(0, 0, 0, 0)
        price_row.setSpacing(6)

        self.price_lbl = QLabel(price)
        self.price_lbl.setObjectName("pricingPrice")

        self.suffix_lbl = QLabel(suffix)
        self.suffix_lbl.setObjectName("pricingSuffix")

        price_row.addWidget(self.price_lbl)
        price_row.addWidget(self.suffix_lbl)
        price_row.addStretch()

        layout.addWidget(self.plan_lbl)
        layout.addLayout(price_row)

        if note_text:
            note_lbl = QLabel(note_text)
            note_lbl.setObjectName("pricingNote")
            layout.addWidget(note_lbl)

        features_layout = QVBoxLayout()
        features_layout.setSpacing(12)

        for feature in features:
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(10)

            icon = QLabel("●")
            icon.setObjectName("featureBullet")

            text = QLabel(feature)
            text.setObjectName("featureText")
            text.setWordWrap(True)

            row.addWidget(icon, 0, Qt.AlignmentFlag.AlignTop)
            row.addWidget(text, 1)
            features_layout.addLayout(row)

        layout.addSpacing(4)
        layout.addLayout(features_layout)
        layout.addStretch()

        self.button = QPushButton(button_text)
        self.button.setProperty("variant", "landing-primary" if highlighted else "landing-plan")
        self.button.clicked.connect(lambda: print(f"{plan_name} selected"))
        layout.addWidget(self.button)


class FaqItem(QFrame):
    def __init__(self, question: str, answer: str, expanded: bool = False, parent: QWidget | None = None):
        super().__init__(parent)
        self.setProperty("variant", "faq-item")

        self._expanded = expanded

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(0)

        self.toggle_btn = QToolButton()
        self.toggle_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle_btn.setText(question)
        self.toggle_btn.setObjectName("faqQuestion")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setChecked(expanded)
        self.toggle_btn.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self.toggle_btn.clicked.connect(self._toggle)

        self.answer_lbl = QLabel(answer)
        self.answer_lbl.setObjectName("faqAnswer")
        self.answer_lbl.setWordWrap(True)
        self.answer_lbl.setVisible(expanded)

        layout.addWidget(self.toggle_btn)
        layout.addSpacing(16)
        layout.addWidget(self.answer_lbl)

    def _toggle(self) -> None:
        self._expanded = self.toggle_btn.isChecked()
        self.toggle_btn.setArrowType(
            Qt.ArrowType.DownArrow if self._expanded else Qt.ArrowType.RightArrow
        )
        self.answer_lbl.setVisible(self._expanded)


class FooterSection(QFrame):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("landingFooter")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(48, 36, 48, 36)
        layout.setSpacing(32)

        brand = QLabel("SEEDVAULT")
        brand.setObjectName("footerBrand")

        links_wrap = QWidget()
        links_layout = QHBoxLayout(links_wrap)
        links_layout.setContentsMargins(0, 0, 0, 0)
        links_layout.setSpacing(24)

        for text in [
            "Terms of Service",
            "Privacy Policy",
            "Whitepaper",
            "Node Status",
            "Support",
        ]:
            btn = QPushButton(text)
            btn.setProperty("variant", "footer-link")
            btn.clicked.connect(lambda _=False, t=text: print(f"{t} clicked"))
            links_layout.addWidget(btn)

        rights = QLabel("© 2024 SEEDVAULT DECENTRALIZED PROTOCOL. ALL RIGHTS RESERVED.")
        rights.setObjectName("footerRights")
        rights.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        layout.addWidget(brand)
        layout.addStretch()
        layout.addWidget(links_wrap)
        layout.addStretch()
        layout.addWidget(rights)


class LandingPageWidget(QWidget):
    def __init__(self, hero_image_path: str | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("landingPageRoot")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        self.nav = LandingNavBar(self)
        self.nav.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.nav.raise_()

        self.hero = HeroSection(image_path=":/images/landing_page.png")

        content_layout.addWidget(self.hero)

        pricing_section = QFrame()
        pricing_section.setObjectName("pricingSection")
        pricing_layout = QVBoxLayout(pricing_section)
        pricing_layout.setContentsMargins(64, 72, 64, 72)
        pricing_layout.setSpacing(40)

        pricing_header = SectionHeader("Select Your Plan", accent=True)
        pricing_layout.addWidget(pricing_header, 0, Qt.AlignmentFlag.AlignHCenter)

        pricing_cards_wrap = QWidget()
        pricing_cards_layout = QGridLayout(pricing_cards_wrap)
        pricing_cards_layout.setContentsMargins(0, 0, 0, 0)
        pricing_cards_layout.setHorizontalSpacing(24)
        pricing_cards_layout.setVerticalSpacing(24)

        monthly_card = PricingCard(
            plan_name="Monthly",
            price="$15",
            suffix="/mo",
            features=[
                "2TB Encrypted Storage",
                "10Gbps Network Speed",
                "Standard Governance Rights",
            ],
            button_text="Select Plan",
            highlighted=False,
        )

        yearly_card = PricingCard(
            plan_name="Yearly",
            price="$120",
            suffix="/year",
            features=[
                "4TB Encrypted Storage",
                "Unlimited Mesh Replication",
                "Priority Node Status",
                "2x Voting Weight",
            ],
            button_text="Get Started",
            highlighted=True,
            badge_text="Best Value",
            note_text="Equivalent to $10/mo",
        )

        pricing_cards_layout.addWidget(monthly_card, 0, 0)
        pricing_cards_layout.addWidget(yearly_card, 0, 1)

        pricing_layout.addWidget(pricing_cards_wrap)

        faq_section = QFrame()
        faq_section.setObjectName("faqSection")
        faq_layout = QVBoxLayout(faq_section)
        faq_layout.setContentsMargins(64, 72, 64, 72)
        faq_layout.setSpacing(28)

        faq_header = SectionHeader(
            "Frequently Asked Questions",
            accent=False,
            label="Knowledge Base",
        )
        faq_layout.addWidget(faq_header)

        faq_items = [
            (
                "How does the seedbox replicate?",
                "SEEDVAULT utilizes a proprietary sharding protocol. When data is uploaded, "
                "it is automatically encrypted and distributed across multiple peer nodes. "
                'The "Self-Replicating" nature ensures that as network demand increases, '
                "active shards clone themselves to under-utilized nodes to maintain 100% availability.",
                True,
            ),
            (
                "How is it democratically controlled?",
                "Governance is baked into the protocol layer. Every subscriber holds a "
                '"Mesh Seed" which acts as a voting share. Proposals regarding protocol '
                "updates, pricing adjustments, and resource allocation are voted on weekly "
                "by the community, ensuring no single entity owns the infrastructure.",
                False,
            ),
            (
                "Where is the data hosted?",
                "Nowhere and everywhere. Unlike traditional seedboxes hosted in centralized "
                "data centers, SEEDVAULT exists on a global mesh of independent nodes. "
                "Your data is never in one physical location, making it resilient to outages and censorship.",
                False,
            ),
            (
                "What happens if a node goes down?",
                "The protocol maintains a persistence ratio. If any shard becomes unavailable "
                "due to a node going offline, the mesh instantly triggers a re-replication event "
                "from other existing copies to restore the redundancy threshold.",
                False,
            ),
        ]

        for question, answer, expanded in faq_items:
            faq_layout.addWidget(FaqItem(question, answer, expanded))

        footer = FooterSection()

        content_layout.addWidget(pricing_section)
        content_layout.addWidget(faq_section)
        content_layout.addWidget(footer)

        self.scroll.setWidget(content)
        root.addWidget(self.scroll)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.nav.setGeometry(0, 0, self.width(), 88)
        self.nav.raise_()
