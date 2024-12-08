from src.stockGroupsService.GroupOver20Years import GroupOver20Years
from src.stockGroupsService.GroupSwiss import GroupSwiss
from src.stockGroupsService.GroupManager import GroupManager
from src.stockGroupsService.GroupSwissOver20Years import GroupSwissOver20Years
from src.stockGroupsService.GroupAmericanOver20Years import GroupAmericanOver20Years
from src.stockGroupsService.GroupSnP500 import GroupSnP500
from src.stockGroupsService.GroupSnP500Over20Years import GroupSnP500Over20Years
from src.stockGroupsService.GroupSnP500NAS100Over20Years import GroupSnP500NAS100Over20Years
from src.stockGroupsService.GroupSnP500FinanTo2011 import GroupSnP500FinanTo2011

def generateGroups():
    dbPath = "src/database"
    groupPath = "src/stockGroups"

    groupClasses = [
        GroupOver20Years(),
        GroupSwiss(),
        GroupSwissOver20Years(),
        GroupAmericanOver20Years(),
        GroupSnP500(),
        GroupSnP500Over20Years(),
        GroupSnP500NAS100Over20Years(),
        GroupSnP500FinanTo2011(),
    ]

    manager = GroupManager(databasePath=dbPath, stockGroupPath=groupPath, groupClasses = groupClasses)
    manager.generateGroups()