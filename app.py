import server
import proc

import streamlit as st # web development
import numpy as np # np mean, np random 
import pandas as pd # read csv, df manipulation
import time # to simulate a real time data, time loop 
import plotly.express as px # interactive charts 
import matplotlib.pyplot as plt
from matplotlib import rcParams
import plotly.graph_objects as go
from PIL import Image
import datetime
import time
import os.path; import os

####>>>Paste Functions Here<<<####                          proc.py

st.set_page_config(
    page_title = '[Room Occupancy:] Real-Time Signal Processing',
    page_icon = '✅',
    layout = 'wide'
)

####>>>Paste 'Real-Time' Database Here<<<####                           server.py

####>>>Paste Server And Processing Code Here<<<#####                            proc.py

# dashboard title
st.title("[Room Occupancy:] Real-Time Signal Processing")


# creating a single-element container.
placeholder = st.empty()
setno = 0
db_localurl = server.dataset_url
# for i in range(4):
#     df = pd.read_csv(db_localurl[i], header=None)
#     print(df.head())
#     print(db_localurl[i])
#     print("::SUCCESS::")
#     print(db_localurl)


# real-time / live feed simulation 
for seconds in range(200): #smth to do with once every 5~6 seconds (doubled)
 
    perso_cnt = 0       

    # run processing mimicking 7 days overwrite
    dataT, df_141complex, array1, array2, perso_cnt, clutter_suppr = proc.main(setno, db_localurl)
    
    with placeholder.container():

        # create placeholder for charts 
 
        print(proc.TP03)

        if perso_cnt >= 2:
            proc.image_show(Image.open("./img/count_b.png"),str(perso_cnt))
                
        elif perso_cnt == 1:
            proc.image_show(Image.open("./img/count_a.png"),"  ")

        else:
            proc.image_show(Image.open("./img/count_z.png"),"  ")
        
            
        # proc.image_show(Image.open("./img/VA 1.png"),"5m x 5m setup")

        st.title("Backend Logs")

        st.markdown("### Detailed UWB Data View")
        st.dataframe(df_141complex)


        st.markdown("### UWB Heatmap ")
        st.write(setno," is " ,db_localurl[setno])
        gofig = go.Figure(data=go.Heatmap(proc.df_to_plotly(df_141complex), colorscale='viridis'))
            #range_x (list of two numbers) – If provided, overrides
            #auto-scaling on the x-axis in cartesian coordinates.
        gofig.update_layout(margin={'t':20,'b':0,'l':0,'r':0})
        st.write(gofig)


        proc.image_show(Image.open("./img/DataflowPicture1.png"),"Explained Dataflow with DataflowPicture1")


        k=3
        st.markdown('### Demo of UWB data > 1st LPF > Cascade 2nd Smoothing')
        ### The following few liners will repeat for each column. Maybe?..
        fig_col2, fig_col3, fig_col4, = st.columns(3)
        fig = plt.figure(figsize=(10, 3))
        
        i=80
        
        with fig_col2:
            st.markdown("UWB Data")
            plt.stem(dataT[i], 'orange', linefmt ='-', markerfmt ='')
            st.pyplot(fig)
            #st.write(sub1)
        with fig_col3:
            st.markdown('1st Filter')
            plt.stem(np.abs(array1[i]), 'green', linefmt ='-', markerfmt ='')
            st.pyplot(fig)
        with fig_col4:
            st.markdown('2nd Filter')
            plt.stem(np.abs(array2[i]),label='temp', linefmt ='-', markerfmt ='')
            st.pyplot(fig)
        
        proc.image_show(Image.open("./img/DataflowPicture2.png"),"Explained Dataflow with DataflowPicture2")


        st.markdown("### Results of Background Removal")
        #4 Result of Cascaded Filter

        #5 Result of Backgnd Removal
                                        #fig = px.density_heatmap(data_frame=clutter_suppr) ###NEEDS EDIT
        
                                        #st.write(fig)
        df_clutter_suppr = pd.DataFrame(clutter_suppr)
        gofig2 = go.Figure(data=go.Heatmap(proc.df_to_plotly(  df_clutter_suppr  ), colorscale='viridis'))
        gofig2.update_layout(margin={'t':20,'b':0,'l':0,'r':0})
        st.write(gofig2)



        plt.close()
        time.sleep(1)
        ###################################################IMF K=3
        cnt = 0
        while(cnt<         np.floor(   len(df_141complex[0])/512   )):
            low = 512*cnt; high = low+512
            proc.image_show("./data/data_LFC/sigall.svg",        "count between {} to %d ==> block signal-512"  .format(str(low).zfill(6)) %high)
            proc.image_show("./data/data_LFC/spectrumall.svg",   "count between {} to %d ==> block signal-log-fft"  .format(str(low).zfill(6)) %high)
            proc.image_show("./data/data_LFC/{}__imf-1.svg" .format(str(low).zfill(6)),      "count between {} to %d ==> IMF 1"  .format(str(low).zfill(6)) %high)
            proc.image_show("./data/data_LFC/{}__imf-2.svg"  .format(str(low).zfill(6)),     "count between {} to %d ==> IMF 2" .format(str(low).zfill(6)) %high)
            proc.image_show("./data/data_LFC/{}__imf-3.svg"  .format(str(low).zfill(6)),     "count between {} to %d ==> IMF 3"  .format(str(low).zfill(6)) %high)
            # image_show(Image.open("./data/data_LFC/{}_imf-%d.svg" .format(str(n).zfill(3)) %(i)),"count between %d to %d" %low %high "==> IMF xx")
            cnt = cnt + 1


        
        
        ###################################################

        # create three columns
        
        kpi1, kpi2, kpi3 = st.columns(3)

        # fill in those three columns with respective metrics or KPIs 
        kpi1.metric(label="Running counter will update every half a minute⏳(fast-forwarded)", value=setno, delta= np.min(dataT[180])+5*np.random.randn() - 10)
        #kpi2.metric(label="Count", value= int(count_married), delta= - 10 + count_married)
        #kpi3.metric(label="A/C Balance ＄", value= f"$ {round(balance,2)} ", delta= - round(balance/count_married) * 100)
        
        time.sleep(1)
        # placeholder.empty()
    

    setno = setno+1 #0 1 2 3 4 5 6
    if setno==3:
        now = datetime.datetime.now()
        setno = 0
        while(1):
            print(now, "!App Displayed!     \n\n\n\n")
            print("Program runs for 3 minutes, to restart, press any key to follow instructions")
            # time.sleep(10)
            break
            
    else: # auto wipe    # wipe away ./data/logs*
        # need something here wipe away data_LFC contents too
        now = datetime.datetime.now()
        print(now, "!App Displayed!     \n\n\n\n")

        myfile="./data/logs"+"0312"+"23.txt"
        if os.path.isfile(myfile):
            os.remove(myfile)

    time.sleep(1)
        
