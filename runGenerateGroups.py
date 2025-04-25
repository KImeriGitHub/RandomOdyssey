import logging
from datetime import datetime
from src.stockGroupsService.GroupOver20Years import GroupOver20Years
from src.stockGroupsService.GroupSwiss import GroupSwiss
from src.stockGroupsService.GroupManager import GroupManager
from src.stockGroupsService.GroupSwissOver20Years import GroupSwissOver20Years
from src.stockGroupsService.GroupSnP500 import GroupSnP500
from src.stockGroupsService.GroupSnP500Over20Years import GroupSnP500Over20Years
from src.stockGroupsService.GroupSnP500NAS100Over20Years import GroupSnP500NAS100Over20Years
from src.stockGroupsService.GroupSnP500FinanTo2011 import GroupSnP500FinanTo2011
from src.stockGroupsService.GroupDebug import GroupDebug
from src.stockGroupsService.GroupFinanTo2011 import GroupFinanTo2011
from src.stockGroupsService.GroupFinanTo2016 import GroupFinanTo2016

formatted_date = datetime.now().strftime("%d%b%y_%H%M").lower()
logging.basicConfig(
    level=logging.INFO, 
    filename=f"output_generategroups_{formatted_date}.log",
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    dbPath = "src/database"
    groupPath = "src/stockGroups"

    groupClasses = [
        GroupOver20Years(),
        GroupSwiss(),
        GroupSwissOver20Years(),
        GroupSnP500(),
        GroupSnP500Over20Years(),
        GroupSnP500NAS100Over20Years(),
        GroupSnP500FinanTo2011(),
        GroupDebug(),
        GroupFinanTo2011(),
        GroupFinanTo2016(),
    ]

    manager = GroupManager(databasePath=dbPath, stockGroupPath=groupPath, groupClasses = groupClasses)
    manager.generateGroups()