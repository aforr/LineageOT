
import pytest

import numpy as np
import ot



class Test_Dependencies():
    """
    Tests for issues with dependencies
    """


    def test_ot_issue_93(self):
        """
        Testing for the issue described at https://github.com/PythonOT/POT/issues/93
        """
        d = ot.emd2([],[],np.ones([292, 3]))
        assert( abs(d-1) < 10**(-8))




