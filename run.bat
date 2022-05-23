FOR /L %%A IN (1,1,5) DO (
  python dataSplit.py
  python main.py
)