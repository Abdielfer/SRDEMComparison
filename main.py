import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import hydra
from hydra import utils 
from omegaconf import DictConfig
import util as U

def saveLogAsText(outPath:str=''):
    '''
    Get the Hydra logs at the current hydra directory: <hydra.core.hydra_config.HydraConfig.get().runtime.output_dir >
    and Save the content as <Logs.txt> in the project working directory. 
    '''
    log_file_name = os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,'main.log')
    print(f"Hydra Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")
    if outPath: 
        log_deposit_txt = outPath
    else:
        log_deposit_txt = os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,'Logs.txt')
   
    print(f"Current WDir : {log_deposit_txt}")
    with open(log_file_name, "r") as log:
        with open(log_deposit_txt, "w") as textLogs:
            for l in log:
               textLogs.write(l) 
    log.close
    textLogs.close
    pass


@hydra.main(version_base=None, config_path=f"hydraConfig", config_name="mainConfig")
def main(cfg: DictConfig):

    U.reportSResDEMComparison(cfg)
    # U.reportSResDEMComparisonSimplified(cfg)
    
    # Get the log file name and save it in text file format(*.txt),  in the same directory as the maps. 
    dem_1 = cfg['dem_1']
    parentDirDEM_1,_,_ = U.get_parenPath_name_ext(dem_1)
    autputTxt = os.path.join(parentDirDEM_1,'Logs.txt') 
    saveLogAsText(autputTxt)
    


if __name__ == "__main__":
    with U.timeit():
        main()  
