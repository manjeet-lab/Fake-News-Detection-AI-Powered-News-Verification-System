"""
=============================================================================
  FRONTEND UI COMPONENTS — MODERN SAAS DASHBOARD RENDERING HELPERS
=============================================================================

Reusable UI rendering helpers for Streamlit frontend.
All components adhere to the dark SaaS aesthetic:
  - Background: #0B0F17
  - Cards: #151C28 / #1A2332
  - Borders: #263143
  - Primary Accent: #7C3AED
  - Success/Danger/Warning: #10B981 / #EF4444 / #F59E0B
"""

import streamlit as st
from svg_icons import get_svg


def is_valid_url(url):
    """
    Validates URL string to prevent broken, fake, or missing links.
    Filters out source name strings, placeholders, and invalid formatted URLs.
    """
    if not url or not isinstance(url, str):
        return False
    u = url.strip()
    if u in ('', '#', 'None', 'null', 'undefined', 'N/A', 'example.com'):
        return False
    
    host_part = u.split('://')[-1].split('/')[0]
    if ' ' in host_part or '%20' in host_part.lower() or ' ' in u:
        return False
        
    if u.startswith(('http://', 'https://')):
        rest = u.split('://', 1)[1].strip()
        domain = rest.split('/')[0]
        return bool(domain and '.' in domain and not domain.startswith('.'))
        
    if '.' in u and not u.startswith('.'):
        domain = u.split('/')[0]
        return bool(domain and '.' in domain)
        
    return False


def render_external_link(label, url):
    """
    Renders an external link button that opens in a NEW browser tab (target="_blank").
    Uses clean SVG ExternalLink icon.
    """
    if not is_valid_url(url):
        st.markdown(f"""
        <div style="font-size: 0.8rem; color: #94a3b8; display: inline-flex; align-items: center; gap: 4px; margin-top: 4px;">
            {get_svg('info', size=14, color='#94a3b8')} Link unavailable
        </div>
        """, unsafe_allow_html=True)
        return

    clean_url = url.strip()
    if not clean_url.startswith(('http://', 'https://')):
        clean_url = 'https://' + clean_url

    clean_label = label.strip() if (label and isinstance(label, str) and label.strip()) else "View Source"
    ext_icon = get_svg('external-link', size=14, color='#a78bfa')

    html = f"""
    <a href="{clean_url}" target="_blank" rel="noopener noreferrer" style="
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(124, 58, 237, 0.12);
        color: #c4b5fd;
        border: 1px solid rgba(139, 92, 246, 0.3);
        padding: 5px 12px;
        border-radius: 8px;
        text-decoration: none;
        font-size: 0.83rem;
        font-weight: 500;
        transition: all 0.2s ease;
        margin-top: 6px;
        margin-bottom: 6px;
    ">
        {ext_icon}
        <span>{clean_label}</span>
    </a>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_status_badge(status):
    """
    Renders a visual status badge with an SVG icon and styled pill container.
    """
    if not status or not isinstance(status, str):
        st_clean = "UNKNOWN"
    else:
        st_clean = status.strip().upper()

    if st_clean in ('SUPPORTED', 'REAL', 'HIGHLY_SUPPORTED', 'CONFIRMED', 'TRUE', 'ONLINE', 'ACTIVE'):
        bg_color = "rgba(16, 185, 129, 0.15)"
        border_color = "rgba(16, 185, 129, 0.4)"
        text_color = "#34d399"
        icon_name = "check-circle"
    elif st_clean in ('CONTRADICTED', 'FAKE', 'FALSE', 'DEBUNKED', 'REFUTED', 'DISPROVEN', 'FAILED'):
        bg_color = "rgba(239, 68, 68, 0.15)"
        border_color = "rgba(239, 68, 68, 0.4)"
        text_color = "#f87171"
        icon_name = "x-circle"
    elif st_clean in ('PARTIALLY_SUPPORTED', 'PARTIALLY_TRUE', 'MIXED', 'READY'):
        bg_color = "rgba(245, 158, 11, 0.15)"
        border_color = "rgba(245, 158, 11, 0.4)"
        text_color = "#fbbf24"
        icon_name = "alert-triangle"
    else:  # UNVERIFIED, UNCERTAIN, UNKNOWN, UNAVAILABLE
        bg_color = "rgba(148, 163, 184, 0.15)"
        border_color = "rgba(148, 163, 184, 0.3)"
        text_color = "#cbd5e1"
        icon_name = "info"

    svg_str = get_svg(icon_name, size=14, color=text_color)

    html = f"""
    <span style="
        background-color: {bg_color};
        color: {text_color};
        border: 1px solid {border_color};
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
        letter-spacing: 0.04em;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    ">
        {svg_str}
        <span>{st_clean}</span>
    </span>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_hero_assessment_card(final_dec):
    """
    Renders the primary Final Assessment verdict card with clean SaaS styling.
    """
    f_label = final_dec.get('final_label') or final_dec.get('final_decision') or 'UNCERTAIN'
    f_level = final_dec.get('evidence_level') or final_dec.get('sub_status') or 'LIMITED EVIDENCE'
    reason = final_dec.get('reason', 'Evidence evaluation completed.')

    if f_label == 'REAL':
        border_col = "#10b981"
        grad_bg = "linear-gradient(135deg, rgba(6, 78, 59, 0.6) 0%, rgba(2, 44, 34, 0.8) 100%)"
        text_col = "#34d399"
        icon_name = "check-circle"
    elif f_label == 'FAKE':
        border_col = "#ef4444"
        grad_bg = "linear-gradient(135deg, rgba(136, 19, 55, 0.6) 0%, rgba(76, 5, 25, 0.8) 100%)"
        text_col = "#f87171"
        icon_name = "x-circle"
    else:  # UNCERTAIN
        border_col = "#f59e0b"
        grad_bg = "linear-gradient(135deg, rgba(120, 53, 15, 0.6) 0%, rgba(69, 26, 3, 0.8) 100%)"
        text_col = "#fbbf24"
        icon_name = "alert-triangle"

    hero_svg = get_svg(icon_name, size=40, color=text_col)

    html = f"""
    <div style="
        background: {grad_bg};
        border: 1px solid {border_col};
        border-radius: 16px;
        padding: 28px 24px;
        text-align: center;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    ">
        <div style="font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.15em; color: #94a3b8; margin-bottom: 8px;">
            FINAL ASSESSMENT
        </div>
        <div style="display: flex; align-items: center; justify-content: center; gap: 12px; margin: 6px 0;">
            {hero_svg}
            <span style="font-size: 2.75rem; font-weight: 800; color: {text_col}; letter-spacing: 0.04em;">{f_label}</span>
        </div>
        <div style="font-size: 0.95rem; font-weight: 600; color: #f8fafc; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 4px;">
            {f_level}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

    # Reason Card
    reason_icon = get_svg('info', size=18, color='#a78bfa')
    reason_html = f"""
    <div style="
        background: #151c28;
        border: 1px solid #263143;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 24px;
    ">
        <div style="display: flex; align-items: center; gap: 8px; color: #a78bfa; font-weight: 600; font-size: 0.95rem; margin-bottom: 8px;">
            {reason_icon}
            <span>Assessment Context & Rationale</span>
        </div>
        <div style="font-size: 0.95rem; line-height: 1.6; color: #e2e8f0;">
            {reason}
        </div>
    </div>
    """
    st.markdown(reason_html, unsafe_allow_html=True)


def render_metric_card(label, value, icon_name="bar-chart", subtext=""):
    """
    Renders a metric card matching dark SaaS theme with SVG icon.
    """
    icon_svg = get_svg(icon_name, size=20, color='#8b5cf6')
    html = f"""
    <div style="
        background: #151c28;
        border: 1px solid #263143;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    ">
        <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 8px;">
            {icon_svg}
        </div>
        <div style="font-size: 1.8rem; font-weight: 800; color: #f8fafc; line-height: 1.2;">
            {value}
        </div>
        <div style="font-size: 0.8rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 4px;">
            {label}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_empty_state(message="No records found.", submessage="There is no data available for this section.", icon_name="info"):
    """
    Renders a clean empty state component with an SVG icon.
    """
    icon_svg = get_svg(icon_name, size=32, color='#64748b')
    html = f"""
    <div style="
        background: #151c28;
        border: 1px dashed #263143;
        border-radius: 12px;
        padding: 32px 20px;
        text-align: center;
        margin: 12px 0;
    ">
        <div style="margin-bottom: 12px;">{icon_svg}</div>
        <div style="font-size: 0.95rem; font-weight: 600; color: #cbd5e1; margin-bottom: 4px;">{message}</div>
        <div style="font-size: 0.85rem; color: #64748b;">{submessage}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
