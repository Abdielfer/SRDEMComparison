# SRDEMComparison
    The goal is to evaluate the DEM enhancement with Super Resolution algorithms. 
    Data:
    cdem: Canadian Digital Elevation Model at 16m resolution.
    srdem: Super Resolution Algorithm Output at 8m resolution. 

    Goals: Evaluate the impact of Super-Resolution algorithms in the cdem transformation. 

        The similarity between the source cdem and srdem will be evaluated through statistics and products derived from the DEM. 

    Proposed measurements of comparison (description): 

        - Elevation statistics' summary comparison:
                Compute mean, std, mode, max and min. Compare the histograms. 
        
        - Slope statistics' summary:
                Compute mean, std, mode, max and min. Compare the histograms.

        - Flow Accumulation summary:
                Compute mean, std, mode, max and min. Compare the histograms.

        - Filled area comparison: 
                Fill the DEM depressions and pits with WhangAndLiu algorithms. Compute the percentage of transformed ares on each dem (cdem_16m and srdem_8m). Compare the percentage of transformed areas. 

        - River network visual comparison:
                Compute strahler order (up to 5th order) and main streams (up to 3rd order). Create maps (Vector) with overlaps of both networks.
            




       

