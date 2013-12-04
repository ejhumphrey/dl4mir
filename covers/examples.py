'''
Created on Oct 8, 2013

@author: ejhumphrey
'''

import os
import ejhumphrey.covers.processes as P

param_dir = "/Volumes/Audio/LargeScaleCoverID/transform_param_files"

pipeline = [P.LogScale(scalar=10.0),
            P.Standardize(param_file=os.path.join(param_dir, "mean_stddev_log1pC10_20131007.pk")),
            P.DotProduct(param_file=os.path.join(param_dir, "random_l2bases_k512.pk")),
            P.RectifiedLinear(theta=0)]


