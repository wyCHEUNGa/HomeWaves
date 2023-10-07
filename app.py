import streamlit as st # web development
import numpy as np # np mean, np random 
import pandas as pd # read csv, df manipulation
import time # to simulate a real time data, time loop 
import plotly.express as px # interactive charts 

from scipy import signal
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import plotly.graph_objects as go

#from scipy.signal import lfilter, lfiltic
import datetime
#import sys

def df_to_plotly(df):
    return {'z': df.values.tolist(),
            'x': df.columns.tolist(),
            'y': df.index.tolist()}

st.set_page_config(
    page_title = '[Room Occupancy:] Real-Time Signal Processing',
    page_icon = '✅',
    layout = 'wide'
)

# read csv from a github repo
#dataset_url = "https://raw.githubusercontent.com/wyCHEUNGa/HomeWaves/main/1rename21.csv"
dataset_url = "https://raw.githubusercontent.com/wyCHEUNGa/HomeWaves/main/raw0glide.csv"
#df_RO = pd.read_csv("https://raw.githubusercontent.com/Lexie88rus/bank-marketing-analysis/master/bank.csv")

# read csv from a URL
@st.cache_data
def get_data() -> pd.DataFrame:
    return pd.read_csv(dataset_url, header=None)

df = get_data()

TP01 = "Initialisaton of Pandas DataFrame completed."
TP03 = "Demo Code Starts Here"
TP12 = "Debugging>>>"

################################################################################################################################################
########################################################################
########################################################################
########################################################################


### Prepare All My Test Buffers ####Tester:      #df_141complex.shape  #series_141
df_141i = pd.DataFrame(index=range(len(df.index)),columns=range(141))
df_141q = pd.DataFrame(index=range(len(df.index)),columns=range(141))
df_141complex = pd.DataFrame(index=range(len(df.index)),columns=range(141))
print(TP01)
#Create sqrt(i^2 + q^2)
for v in range(141):
    df_141i[v] = df[v] ###+df[i+141]
for v in range(141):
    df_141q[v] = df[v+141]
for v in range(141):
    df_141complex[v] = np.sqrt(   np.power(df_141i[v],2) +  np.power(df_141q[v],2)   )
print(70)
#Transposed for signal processing
dataT=df_141complex.transpose()

################################################################################################################################################

### Prior starting signal processing, the order of windows is chosen as 26 and 50 resp.
rcParams['figure.figsize'] = 5.7,3.27
#Window shape Hamming, while the LPF filter
# freqy above 250Hz
numtaps = 26
f2 = 250/1000 #end of notch
h = signal.firwin(numtaps+1, f2*2,pass_zero='lowpass',window='hamming')
#plt.stem(fftpack.fftfreq(h.size),     np.abs(fftpack.fft(h)),  linefmt ='--', markerfmt ='')

numtaps = 50
f2 = 100/1000
h50 = signal.firwin(numtaps+1, f2*2)
#plt.xlim(0,0.6)

###Use of LPF n Smooth Filter  (17 Sep 23)
from numpy.fft import fft, ifft,fftfreq, fftshift
from scipy import fftpack

s = (len(df_141complex[0]),141)

array1=np.zeros(s, dtype=complex)

array2=np.zeros(s, dtype=complex)

#M = 20
#coef = np.hamming(M+1)
#coef = signal.windows.hamming(M+1)
start_dist_of_interest = 0



### Starting to process DataT
for i in range(0,len(df_141complex[0])):  
    filtered_24 = signal.lfilter(h,1,dataT[start_dist_of_interest]) 
    topl=np.copy(filtered_24[:12])
    for i in range(0,141):
        if i<129: #0 to 128
            filtered_24[i]=filtered_24[i+12]
        else:     #129 130 131 132 133 4 5 6 7 8 9 140
            filtered_24[i]=topl[i-129]
    temp=fftpack.fft(filtered_24)
    filtered_24b = fftpack.ifft(temp) #you'll want to print this for proof #add into 2D array
    
    ######################
    #Cascaded a second filter
    #dataT_sfil= np.convolve(fftpack.ifft(temp),coef,mode='same')           #add into 2D arr
    dataT_sfil = signal.lfilter(h50,1,filtered_24b)
    topr=np.copy(dataT_sfil[:25])
    for i in range(0,141):
        if i<116: #0 to 115
            dataT_sfil[i]=dataT_sfil[i+24]
        else:     #116 117 ... 131 132 133 4 5 6 7 8 9 140
            dataT_sfil[i]=topr[i-116]
    
    array1[start_dist_of_interest]=filtered_24b
    array2[start_dist_of_interest]=dataT_sfil
    
    start_dist_of_interest=start_dist_of_interest+1 #until 0,1131-1,1131

print('Done Signal Proc, 3 charts will be displayed.')

################################################################################################################################################

### Starting Background Noise Removal
beta=0.97
k=7

s = (len(df_141complex[0]),len(array2[0]))
B2=np.zeros(s)
clutter_suppr=np.zeros(s)

for k in range(0, len(df_141complex[0])):
    for i in range(0, len(dataT[0])):
        if i == 0:
            B2[k][i] = (1-beta)*np.abs(array2[k][i])
        else:
            B2[k][i] = (1-beta)*np.abs(array2[k][i]) + beta*B2[k][i-1]
        
        clutter_suppr[k][i] = np.abs(array2[k][i]) - B2[k][i]
print("Done Clutter Shift, 2nd heatmap in line will be displayed.")

################################################################################################################################################

### Starting User Identification

#output b_s and b and comparision statement
###Total: 1131 | Strategically Take x[0-9][0] | Take x[0-9][1] | ...| Take x[0-9][140]
###Created by the above strategic, a 9x141 short observation. There r 4 short in 1 long observ
s = (141,10); b_s=np.zeros(s)
s = (141,40); b_l=np.zeros(s)
s = (10); tem=np.zeros(s)
s = (40); longobsvtem=np.zeros(s)
c = 1.2
cnt = 0

#Update fastdetected database, How to use:
#Mark the binary outcome of fast movement >Take the 'low' indices > Put
#inside a queue >This in turn will be used by PEAK DETECTION
fastdetected = 0

while(cnt<np.floor(len(df_141complex[0])/40)):

    ####For the 1st b_l block... get b_l block for [0-40)
    low=40*cnt; high=low+40 #40 will be exclusive
    ptr = np.random.randint(low,high-9)            #randomise any 10 data for bs. Param 2 exclusive.
    for j in range(0,141):
        for i in range(ptr,ptr+10):      #any 10 between [0-30] 30 incluc.
            
            tem[i-ptr]=clutter_suppr[i][j]      #clutter_suppr[10][0] to clutter_suppr[19][0]
        
        b_s[j][:]=np.abs(fftpack.fft(tem))
    print('TP11: ptr is ', ptr, ' cnt is ', cnt)

    cnt=cnt+1
    ########################################################

    ###30 31        32 33      34 35           36 37           38 39###
    ####For the next b_l block.
    #choose another [40-80) by using cnt++; low=cnt*40

    #Code for b_s signalSTR
    #Compare bssignalSTR > c*bl c*signalSTR
    #Yes then discard  #No then further use peak detection
    signalSTR = np.max(b_s)/(np.sum(b_s)/(10*141))

    for j in range(0,141):
        for i in range(low,high):      #  [40-80)
            
            longobsvtem[i-low]=clutter_suppr[i][j]      #clutter_suppr[40][0] to clutter_suppr[79][0]
                                                    #longobsvtem[40]
                                                    #i = [0:40)
                                                    #i = [1080:1120)
                                                    #i = [1120:~nvm]
        
        b_l[j][:]=np.abs(fftpack.fft(longobsvtem))
        
    signalSTRL = np.max(b_l)/(np.sum(b_l)/(10*141))
    if signalSTR>(c*signalSTRL):
        print(signalSTR,signalSTRL*c,'To Be Discarded.')
        fastdetected = 1
    else:
        print(signalSTR,signalSTRL*c,'No Fast Movement Detected, Proceed.')
        fastdetected = 0

########################################################################
########################################################################
########################################################################
################################################################################################################################################

# dashboard title

st.title("[Room Occupancy:] Real-Time Signal Processing")


# creating a single-element container.
placeholder = st.empty()


# near real-time / live feed simulation 

for seconds in range(190): #smth to do with once every 5~6 seconds
#while True: 
    
    #These are some tricks
    
    #df_RO['age_new'] = df_RO['age'] * np.random.choice(range(1,5)) #Creative but what's the point x1 to x4
    #df_RO['balance_new'] = df_RO['balance'] * np.random.choice(range(1,5)) #likewise, no point

    # creating KPIs. These are some tricks
    
    #avg_age = np.mean(df_RO['age_new']) 

    #count_married = int(df_RO[(df_RO["marital"]=='married')]['marital'].count() + np.random.choice(range(1,30)))
    
    #balance = np.mean(df_RO['balance_new'])

    with placeholder.container():

        # create placeholder for charts 


        st.markdown("### Detailed Data View")
        st.dataframe(df_141complex)


        st.markdown("### UWB Heatmap")
        gofig = go.Figure(data=go.Heatmap(df_to_plotly(df_141complex), colorscale='viridis'))
            #range_x (list of two numbers) – If provided, overrides
            #auto-scaling on the x-axis in cartesian coordinates.
        gofig.update_layout(margin={'t':20,'b':0,'l':0,'r':0})
        #figline.update_layout(margin={'t':0,'b':0,'l':0,'r':0})
        st.write(gofig)


        k=3
        st.markdown('### Demo of UWB data > 1st LPF > Cascade 2nd Smoothing')
        ### The following few liners will repeat for each column. Maybe?..
        fig_col2, fig_col3, fig_col4, = st.columns(3)
        fig = plt.figure(figsize=(10, 8))
        
        i=80
        print(TP03)
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
        

        st.markdown("### Results of Background Removal")
        #4 Result of Cascaded Filter

        #5 Result of Backgnd Removal
                                        #fig = px.density_heatmap(data_frame=clutter_suppr) ###NEEDS EDIT
        
                                        #st.write(fig)
        df_clutter_suppr = pd.DataFrame(clutter_suppr)
        gofig2 = go.Figure(data=go.Heatmap(df_to_plotly(  df_clutter_suppr  ), colorscale='viridis'))
        gofig2.update_layout(margin={'t':20,'b':0,'l':0,'r':0})
        st.write(gofig2)



        plt.close()
        time.sleep(1)
        ###################################################

        # create three columns
        
        kpi1, kpi2, kpi3 = st.columns(3)

        # fill in those three columns with respective metrics or KPIs 
        kpi1.metric(label="df_141complex min value⏳", value=np.min(dataT[180])+5*np.random.randn(), delta= np.min(dataT[180])+5*np.random.randn() - 10)
        #kpi2.metric(label="Married Count 💍", value= int(count_married), delta= - 10 + count_married)
        #kpi3.metric(label="A/C Balance ＄", value= f"$ {round(balance,2)} ", delta= - round(balance/count_married) * 100)
        
        now = datetime.datetime.now()
        print(now, "!EOL!     \n\n\n\n")
    #placeholder.empty()