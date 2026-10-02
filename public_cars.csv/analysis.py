from pathlib import Path
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
BASE=Path(__file__).parent; DATA=BASE/'data'; USD_TO_INR=83.5; ANALYSIS_YEAR=pd.Timestamp.today().year
FEATURES=['Brand','Model','Transmission','Color','Mileage_km','Year_Manufactured','Fuel_Type','Has_Gas_System','Engine_Type','Engine_Capacity_L','Body_Type','Has_Warranty','Ownership_Status','Drivetrain']+[f'Feature_{i:02d}' for i in range(1,11)]+['Days_Listed','Car_Age','Mileage_Thousands']
def load_clean_data(path=DATA/'cleaned_cars.csv'): return pd.read_csv(path)
def train_models(df):
 X,y=df[FEATURES],df.Price_INR; Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
 cat=X.select_dtypes(include=['object','string']).columns.tolist(); nums=[c for c in X if c not in cat]
 pre=ColumnTransformer([('num',SimpleImputer(strategy='median'),nums),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('ohe',OneHotEncoder(handle_unknown='ignore'))]),cat)])
 estimators={'Linear Regression':LinearRegression(),'Random Forest':RandomForestRegressor(n_estimators=100,random_state=42,n_jobs=-1,min_samples_leaf=2),'Gradient Boosting':GradientBoostingRegressor(random_state=42,n_estimators=100)}; rows=[]; fitted={}
 for name,est in estimators.items():
  pipe=Pipeline([('preprocessor',pre),('model',est)]); pipe.fit(Xtr,ytr); p=pipe.predict(Xte); fitted[name]=pipe; rows.append({'Model':name,'MAE':mean_absolute_error(yte,p),'RMSE':mean_squared_error(yte,p)**.5,'R2':r2_score(yte,p)})
 m=pd.DataFrame(rows).sort_values('RMSE').reset_index(drop=True); best=fitted[m.loc[0,'Model']]; return best,m,Xte,yte,best.predict(Xte)
def predict_car_price(details,model):
 row=pd.DataFrame([details]); row['Car_Age']=ANALYSIS_YEAR-row.Year_Manufactured; row['Mileage_Thousands']=row.Mileage_km/1000; return float(model.predict(row)[0])
