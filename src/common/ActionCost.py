class ActionCost:
    def __init__(self):
        self.isNoCost = False

    def buy(self, price:float):
        if self.isNoCost:
            return 0
        
        stempelGebuehr = 0.15 * 0.01
        commission = 0.3 * 0.01
        total_portion = stempelGebuehr + commission

        return price * total_portion 
    
    def sell(self, price:float):
        return self.buy(price)