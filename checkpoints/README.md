# Training checkpoint folder

Populated by training method, stores model architecture in JSON and weights in HDF5 format.

Directory format: &lt;model&gt;\_&lt;scale&gt;\_&lt;resolution&gt;  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;/&lt;epoch&gt;.h5: Model weights at particular training epoch  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;/arch.json: Model archtecture  
