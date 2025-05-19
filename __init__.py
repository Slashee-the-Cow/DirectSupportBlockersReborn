# Direct Support Blockers Rebron
#
# Copyright 2025- Slashee the Cow

from . import DirectSupportBlockersReborn
from UM.i18n import i18nCatalog

i18n_catalog = i18nCatalog("directsupportblockers")

def getMetaData():
    """Tell Cura about the shiny plugin"""
    return {
        "tool": {
            "name": i18n_catalog.i18nc("@metadata:name", "Direct Support Blockers Reborn"),
            "description": i18n_catalog.i18nc("@metadata:description", "Create support blockers other than the default box easily"),
            "icon": "dsrb_icon.svg",
            "tool_panel": "qml/dsrb_panel.qml",
            "weight": 7
        }
    }

def register(app):
    """Register tool for Cura to see"""
    return {"tool": DirectSupportBlockersReborn.DirectSupportBlockersReborn()}
