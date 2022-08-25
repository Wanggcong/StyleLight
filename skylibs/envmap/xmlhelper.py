import xml.etree.ElementTree as ET


class EnvmapXMLParser:
    """
    Parser for the metadata file ( filename.meta.xml ).
    """
    def __init__(self, filename):
        self.tree = ET.parse(filename)
        self.root = self.tree.getroot()

    def _getFirstChildTag(self, tag):
        for elem in self.root:
            if elem.tag == tag:
                return elem.attrib

    def _getAttrib(self, node, attribute, default=None):
        if node:
            return node.get(attribute, default)
        return default

    def getFormat(self):
        """Returns the format of the environment map."""
        node = self._getFirstChildTag('data')
        return self._getAttrib(node, 'format', 'Unknown')

    def getDate(self):
        """Returns the date of the environment mapin dict format."""
        return self._getFirstChildTag('date')

    def getExposure(self):
        """Returns the exposure of the environment map in EV."""
        node = self._getFirstChildTag('exposure')
        return self._getAttrib(node, 'EV')
