import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import hydra 
from omegaconf import DictConfig
import util as U
import qgisTools as QT


@hydra.main(version_base=None, config_path=f"hydraConfig", config_name="mainConfig")
def main(cfg: DictConfig):
    shp1 = r'C:\Users\abfernan\CrossCanFloodMapping\SResDEM\Data\LayOuttest\cdem_Rivernet3rdOrder.shp'
    shp2 = r'C:\Users\abfernan\CrossCanFloodMapping\SResDEM\Data\LayOuttest\srdem_Rivernet3rdOrder.shp'
    shpOut = r'C:\Users\abfernan\CrossCanFloodMapping\SResDEM\Data\LayOuttest\QC_Quebec_river3rdOrder_1.png'


    QT.overlap_vectors(shp1,shp2,shpOut)
    
    # QT.create_layout(shp1,shp2,shpOut)
    
if __name__ == "__main__":
    with U.timeit():
        main()  
