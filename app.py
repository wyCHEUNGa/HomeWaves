import streamlit as st # web development
import numpy as np # np mean, np random 
import pandas as pd # read csv, df manipulation
import time # to simulate a real time data, time loop 
import plotly.express as px # interactive charts 

#from numpy.fft import fft, ifft,fftfreq, fftshift
from scipy import fftpack
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib import rcParams
#import seaborn as sns
import plotly.graph_objects as go

#from scipy.signal import lfilter, lfiltic
from PIL import Image
import datetime
#import sys
#import os

###Credits Sakari Cajanus 1 Mar 2020 on https://stackoverflow.com/questions/49827096/generating-a-plotly-heat-map-from-a-pandas-pivot-table
def df_to_plotly(df):
    return {'z': df.values.tolist(),
            'x': df.columns.tolist(),
            'y': df.index.tolist()}

def update_logs091023(low, high, fd, filename):
    with open(filename, 'a') as f:
        if fd == 1:
            f.write(str(low) + "\t" + str(high-1) + "\t" + str(fd) + "\t" + "No file\n")
        else:
            f.write(str(low) + "\t" + str(high-1) + "\t" + str(fd) + "\t" + "See ---.txt\n")

def peak_detected(b_l, low, high, fd):
    temp=[]

    if fd == 1:
        temp[:40] = np.zeros(40)

    else:
    #{'1':low,'2':high-1,'3':fd,'4':"add results of sliding window"}

        for h in range(0,40):
            with open("./data/b_l_range"+str(low)+"_to_"+str(high-1)+".txt"    , "a") as branch_file:

                for w in range(0,141):
                    #TESTER w=5; h=20

                    if h==0:
                        temp.append(0)

                    elif h==1:
                        temp.append(0)

                    elif (1<h<38):
                        if w <= 3: 
                            temp.append(0)
                        elif (3< w <137):
                            ### Starting Peak Detection. Creation of 9x5 sliding window. 
                            th_motion = (
                            b_l[w-4][h-2]+
                            b_l[w-3][h-2]+
                            b_l[w-4][h-1]+
                            b_l[w-3][h-1]+
                            b_l[w-4][h-0]+
                            b_l[w-3][h-0]+
                            b_l[w-4][h+1]+
                            b_l[w-3][h+1]+
                            b_l[w-4][h+2]+
                            b_l[w-3][h+2]+

                            b_l[w-2][h-2]+
                            b_l[w-1][h-2]+
                            b_l[w-0][h-2]+
                            b_l[w+1][h-2]+
                            b_l[w+2][h-2]+

                            b_l[w+4][h-2]+
                            b_l[w+3][h-2]+
                            b_l[w+4][h-1]+
                            b_l[w+3][h-1]+
                            b_l[w+4][h-0]+
                            b_l[w+3][h-0]+
                            b_l[w+4][h+1]+
                            b_l[w+3][h+1]+
                            b_l[w+4][h+2]+
                            b_l[w+3][h+2]+

                            b_l[w-2][h+2]+
                            b_l[w-1][h+2]+
                            b_l[w-0][h+2]+
                            b_l[w+1][h+2]+
                            b_l[w+2][h+2])/30

                            #print("b_l["    ,w,   "]["   ,h,   "] is ",
                            #      np.round(b_l[w][h],6), " \t 1.5*noise threshold is ", np.round(1.5*th_motion,6))

                            branch_file.write("b_l["  +  str(w)  +  "]["  +  str(h)  +  "] is "  +
                                str(np.round(b_l[w][h],6))  +  " \t 1.5*noise threshold is "  +  str(np.round(1.5*th_motion,6))+"\n")
                            if b_l[w][h]>=1.5*th_motion:
                                branch_file.write("Yes human detected.\n")
                                temp.append(1)
                            else:
                                branch_file.write("Move to next.\n")
                                temp.append(0)
                        else: #w gt or equal to 137
                            temp.append(0)

                    elif h==38:
                        temp.append(0)

                    elif h==39:
                        temp.append(0)
        
    update_logs091023(low, high, fd, "./data/logs"+DDMM+"23.txt")
    
    return temp

def image_show(image,caption):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(' ')

    with col2:
        st.image(image, caption=caption)

    with col3:
        st.write(' ')

st.set_page_config(
    page_title = '[Room Occupancy:] Real-Time Signal Processing',
    page_icon = '✅',
    layout = 'wide'
)

# read csv from a github repo
#dataset_url = "https://raw.githubusercontent.com/wyCHEUNGa/HomeWaves/main/1rename21.csv"
dataset_url = "https://raw.githubusercontent.com/wyCHEUNGa/HomeWaves/main/test2.csv"
#df_RO = pd.read_csv("https://raw.githubusercontent.com/Lexie88rus/bank-marketing-analysis/master/bank.csv")

# read csv from a URL
@st.cache_data
def get_data() -> pd.DataFrame:
    return pd.read_csv(dataset_url, header=None)
#df = get_data()
df = pd.read_csv(dataset_url, header=None)

DDMM = "1510"  ### Update Correct Today's Date ###
TP01 = "Program Loaded"
TP02 = "Initialisaton of Pandas DataFrame completed."
TP03 = "Demo Code Starts Here"
TP12 = "Debugging>>>"

################################################################################################################################################
########################################################################
########################################################################
########################################################################

now = datetime.datetime.now()
try:
    f = open("./data/logs"+DDMM+"23.txt","x")
    f.write("Computer Time is" + str(now) + "\n")
    f.close()
except OSError as e:
    print(TP12, " Try deleting files in ./data folder and refresh programme")
    print("\n\nReady, Set, Go!")
    exit()
else:
    print(TP01)

### Prepare All My Test Buffers ####Tester:      #df_141complex.shape  #series_141
df_141i = pd.DataFrame(index=range(len(df.index)),columns=range(141))
df_141q = pd.DataFrame(index=range(len(df.index)),columns=range(141))
df_141complex = pd.DataFrame(index=range(len(df.index)),columns=range(141))
print(TP02)
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
numtaps = 26 #page 70:09
f2 = 250/1000 #end of notch
h = signal.firwin(numtaps+1, f2*2,pass_zero='lowpass',window='hamming')
#plt.stem(fftpack.fftfreq(h.size),     np.abs(fftpack.fft(h)),  linefmt ='--', markerfmt ='')

numtaps = 50 #page 70:09
f2 = 100/1000
h50 = signal.firwin(numtaps+1, f2*2)
#plt.xlim(0,0.6)

###Use of LPF n Smooth Filter  (17 Sep 23)
s = (len(df_141complex[0]),141)

array1=np.zeros(s, dtype=complex)

array2=np.zeros(s, dtype=complex)

#M = 50
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
beta=0.97 #page 70:10 #𝑐𝑘(𝑡) = 𝛽𝑐𝑘−1(𝑡) + (1 − 𝛽)𝑟𝑘(𝑡)
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
c = 1.2 #page 70:10
cnt = 71

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

    #For 1x single b_l block, finding out FFT is useable or not useable
    signalSTR = np.max(b_s)/(np.sum(b_s)/(10*141))

    for j in range(0,141):
        for i in range(low,high):      #  [40-80)
            
            longobsvtem[i-low]=clutter_suppr[i][j]      #clutter_suppr[40][0] to clutter_suppr[79][0]
                                                    #longobsvtem[40]
                                                    #i = [0:40)
                                                    #i = [1080:1120)
                                                    #i = [1120:~not computerised]
        
        b_l[j][:]=np.abs(fftpack.fft(longobsvtem))
        
    signalSTRL = np.max(b_l)/(np.sum(b_l)/(40*141))
    if signalSTR>(c*signalSTRL):
        print(signalSTR,signalSTRL*c,'To Be Discarded.') #page 70:10
        fastdetected = 1
        
    else:
        print(signalSTR,signalSTRL*c,'No Fast Movement Detected, Proceed.')
        fastdetected = 0

    ############################################################################################################################################
    
    #Continued, so if FFT is useable, find peak amplitude

    flags_if = peak_detected(b_l, low, high, fastdetected)
    for i in range(len(flags_if)):
        if i%141 == 140:
            print(flags_if[i])
        else:
            print(flags_if[i],end="")
    print("\n\n")
    print("Done Selecting Peaks, Logs Available. ")

################################################################################################################################################

#### Starting "Vibration Decomposition via Variational Mode Decomposition" -- Multi-Sequence VMD Algorithm

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


        st.markdown("### Detailed UWB Data View")
        st.dataframe(df_141complex)


        st.markdown("### UWB Heatmap")
        gofig = go.Figure(data=go.Heatmap(df_to_plotly(df_141complex), colorscale='viridis'))
            #range_x (list of two numbers) – If provided, overrides
            #auto-scaling on the x-axis in cartesian coordinates.
        gofig.update_layout(margin={'t':20,'b':0,'l':0,'r':0})
        st.write(gofig)


        image_show(Image.open("./img/DataflowPicture1.png"),"Explained Dataflow with DataflowPicture1")


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
        

        image_show(Image.open("./img/DataflowPicture2.png"),"Explained Dataflow with DataflowPicture2")


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
        kpi1.metric(label="Test update every 5~6 seconds⏳", value=np.min(dataT[180])+5*np.random.randn(), delta= np.min(dataT[180])+5*np.random.randn() - 10)
        #kpi2.metric(label="Married Count 💍", value= int(count_married), delta= - 10 + count_married)
        #kpi3.metric(label="A/C Balance ＄", value= f"$ {round(balance,2)} ", delta= - round(balance/count_married) * 100)
        
        now = datetime.datetime.now()
        print(now, "!EOL!     \n\n\n\n")
    #placeholder.empty()