import streamlit as st, pandas as pd, plotly.express as px, plotly.graph_objects as go
from analysis import *
st.set_page_config(page_title='Car Resale Price Intelligence',page_icon='🚗',layout='wide')
@st.cache_data
def data(): return load_clean_data()
@st.cache_resource
def fitted(d): return train_models(d)
df=data(); model,metrics,Xte,yte,pred=fitted(df)
st.title('🚗 Car Resale Price Intelligence Dashboard'); st.caption(f'1 USD = ₹{USD_TO_INR:.2f} | Analysis year: {ANALYSIS_YEAR}')
with st.sidebar:
 brands=st.multiselect('Brand',sorted(df.Brand.unique())); fuels=st.multiselect('Fuel Type',sorted(df.Fuel_Type.unique())); trans=st.multiselect('Transmission',sorted(df.Transmission.unique())); bodies=st.multiselect('Body Type',sorted(df.Body_Type.unique())); yr=st.slider('Year',int(df.Year_Manufactured.min()),int(df.Year_Manufactured.max()),(int(df.Year_Manufactured.min()),int(df.Year_Manufactured.max())))
f=df[df.Year_Manufactured.between(*yr)];
for c,v in [('Brand',brands),('Fuel_Type',fuels),('Transmission',trans),('Body_Type',bodies)]:
 if v: f=f[f[c].isin(v)]
t=st.tabs(['Overview','Market','Analytics','Prediction','Models','Quality'])
with t[0]:
 a,b,c,d,e=st.columns(5); a.metric('Total Cars',f'{len(f):,}'); b.metric('Average Price',f'₹{f.Price_Lakh.mean():.2f} L'); c.metric('Median Price',f'₹{f.Price_Lakh.median():.2f} L'); d.metric('Average Age',f'{f.Car_Age.mean():.1f} y'); e.metric('Average Mileage',f'{f.Mileage_km.mean():,.0f} km'); st.plotly_chart(px.histogram(f,x='Price_Lakh',nbins=40,title='Price distribution'),use_container_width=True); st.plotly_chart(px.bar(f.Brand.value_counts().head(15).reset_index(name='Cars'),x='Brand',y='Cars',title='Brand distribution'),use_container_width=True)
with t[1]:
 for c in ['Brand','Model','Fuel_Type','Transmission','Body_Type','Drivetrain','Has_Warranty']:
  q=f.groupby(c).agg(Median_Price=('Price_Lakh','median'),Mean_Price=('Price_Lakh','mean'),Cars=('Price_Lakh','size')).reset_index(); q=q[q.Cars>=5].sort_values('Cars',ascending=False).head(20); st.plotly_chart(px.bar(q,x=c,y='Median_Price',color='Cars',hover_data=['Mean_Price','Cars'],title=f'{c} vs price'),use_container_width=True)
with t[2]:
 for x in ['Mileage_km','Car_Age','Engine_Capacity_L','Year_Manufactured','Days_Listed']: st.plotly_chart(px.scatter(f,x=x,y='Price_Lakh',color='Fuel_Type',trendline='ols',hover_data=['Brand','Model'],title=f'Price vs {x}'),use_container_width=True)
 st.plotly_chart(px.imshow(f[['Price_INR','Mileage_km','Car_Age','Engine_Capacity_L','Year_Manufactured','Days_Listed']].corr(),text_auto='.2f',title='Correlation matrix'),use_container_width=True)
with t[3]:
 st.subheader('Estimate resale price')
 with st.form('p'):
  d={}; cols=st.columns(3)
  for i,c in enumerate(['Brand','Model','Transmission','Color','Fuel_Type','Engine_Type','Body_Type','Ownership_Status','Drivetrain']): d[c]=cols[i%3].selectbox(c,sorted(df[c].unique()))
  d['Mileage_km']=cols[0].number_input('Mileage km',0,2000000,100000); d['Year_Manufactured']=cols[1].number_input('Year',int(df.Year_Manufactured.min()),ANALYSIS_YEAR,2015); d['Engine_Capacity_L']=cols[2].number_input('Engine L',.1,10.,1.6); d['Days_Listed']=cols[0].number_input('Days listed',0,5000,30); d['Has_Gas_System']=cols[1].checkbox('Gas system'); d['Has_Warranty']=cols[2].checkbox('Warranty')
  for i in range(1,11): d[f'Feature_{i:02d}']=st.checkbox(f'Feature_{i:02d}',key=i)
  if st.form_submit_button('Predict'): p=predict_car_price(d,model); st.success(f'Estimated Resale Price: ₹{p/1e5:.2f} Lakhs (₹{p:,.0f})'); st.info('Estimate only; no artificial confidence percentage is provided.')
with t[4]:
 st.dataframe(metrics.style.format({'MAE':'₹{:,.0f}','RMSE':'₹{:,.0f}','R2':'{:.3f}'}),use_container_width=True); z=pd.DataFrame({'Actual':yte,'Predicted':pred}); fig=px.scatter(z,x='Actual',y='Predicted',title='Actual vs predicted'); lo=min(z.min()); hi=max(z.max()); fig.add_trace(go.Scatter(x=[lo,hi],y=[lo,hi],mode='lines',name='Perfect')); st.plotly_chart(fig,use_container_width=True); st.plotly_chart(px.histogram(x=yte-pred,title='Residual distribution'),use_container_width=True)
with t[5]: st.write({'Original rows':30824,'Duplicate rows removed':36,'Final rows':len(df),'Missing after cleaning':int(df.isna().sum().sum()),'Undocumented features':'Feature_01–Feature_10'}); st.dataframe(pd.DataFrame({'Column':df.columns,'Dtype':df.dtypes.astype(str)}),use_container_width=True)
