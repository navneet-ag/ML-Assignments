class MyNeuralNetwork():
    """
    My implementation of a Neural Network Classifier.
    """

    acti_fns = ['relu', 'sigmoid', 'linear', 'tanh', 'softmax']
    weight_inits = ['zero', 'random', 'normal']

    def __init__(self, n_layers, layer_sizes, activation, learning_rate, weight_init, batch_size, num_epochs):
        """
        Initializing a new MyNeuralNetwork object

        Parameters
        ----------
        n_layers : int value specifying the number of layers

        layer_sizes : integer array of size n_layers specifying the number of nodes in each layer

        activation : string specifying the activation function to be used
                     possible inputs: relu, sigmoid, linear, tanh

        learning_rate : float value specifying the learning rate to be used

        weight_init : string specifying the weight initialization function to be used
                      possible inputs: zero, random, normal

        batch_size : int value specifying the batch size to be used

        num_epochs : int value specifying the number of epochs to be used
        """

        if activation not in self.acti_fns:
            raise Exception('Incorrect Activation Function')

        if weight_init not in self.weight_inits:
            raise Exception('Incorrect Weight Initialization Function')
        
        if n_layers != len(layer_sizes):
            print(n_layers,len(layer_sizes))
            raise Exception('Incorrect Layer or Layer size description')            
        
        self.n_layers    = n_layers 
        self.layer_sizes = layer_sizes
        self.activation  = activation
        self.weight_init = weight_init
        self.batch_size  = batch_size
        self.num_epochs  = num_epochs
        self.learning_rate = learning_rate
        
        self.weight = {}
        self.bias    = {}
        
        for i in range(1,self.n_layers):
            
            shape_wt   = (layer_sizes[i-1],layer_sizes[i])
            shape_bias = (1,layer_sizes[i])

            self.bias[i]   = np.zeros(shape_bias)
            
            if(self.weight_init == "zero" ):
                self.weight[i] = self.zero_init(shape_wt)

            elif(self.weight_init == "random" ):
                self.weight[i] = self.random_init(shape_wt)
            
            else:
                self.weight[i] = self.normal_init(shape_wt)
                

    
    def relu(self, X):
        """
        Calculating the ReLU activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        X_mod = np.array(X,copy = True)
        X_mod[X_mod<0] = 0
        return X_mod

    def relu_grad(self, X):
        """
        Calculating the gradient of ReLU activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        X_mod = np.array(X,copy = True)
        X_mod[X_mod >= 0] = 1
        X_mod[X_mod < 0] = 0        
        return X_mod

    def sigmoid(self, X):
        """
        Calculating the Sigmoid activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return 1/(1+np.exp(-X))

    def sigmoid_grad(self, X):
        """
        Calculating the gradient of Sigmoid activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        sigmoid_x = self.sigmoid(X) 
        return (1-sigmoid_x)*sigmoid_x

    def linear(self, X):
        """
        Calculating the Linear activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return X
        
    def linear_grad(self, X):
        """
        Calculating the gradient of Linear activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return np.ones(X.shape)

    def tanh(self, X):
        """
        Calculating the Tanh activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return np.tanh(X)

    def tanh_grad(self, X):
        """
        Calculating the gradient of Tanh activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 
        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        tanh_x = self.tanh(X)
        
        return 1 - (tanh_x*tanh_x)

    def softmax(self, X):
        """
        Calculating the ReLU activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        numerator   = np.exp(X)
        denominator = np.sum(numerator,axis=1,keepdims=True) 
        return numerator/denominator
        

    def softmax_grad(self, X):
        """
        Calculating the gradient of Softmax activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        # if softmax gives p_i = e^z_i / sum (e^z_j for all j)
        #The gradient of softmax (dp_i/dz_j) is p_i(1-p_i) if j=i else its -p_i.p_j
        #but we can directly use cross-entropy's loss derivative wrt softmax which is y_pred - y_actual
        return None

    def zero_init(self, shape):
        """
        Calculating the initial weights after Zero Activation for a particular layer

        Parameters
        ----------
        shape : tuple specifying the shape of the layer for which weights have to be generated 

        Returns
        -------
        weight : 2-dimensional numpy array which contains the initial weights for the requested layer
        """
        return np.zeros(shape)

    def random_init(self, shape):
        """
        Calculating the initial weights after Random Activation for a particular layer

        Parameters
        ----------
        shape : tuple specifying the shape of the layer for which weights have to be generated 

        Returns
        -------
        weight : 2-dimensional numpy array which contains the initial weights for the requested layer
        """
        return 0.01 * np.random.rand(shape[0],shape[1])

    def normal_init(self, shape):
        """
        Calculating the initial weights after Normal(0,1) Activation for a particular layer

        Parameters
        ----------
        shape : tuple specifying the shape of the layer for which weights have to be generated 

        Returns
        -------
        weight : 2-dimensional numpy array which contains the initial weights for the requested layer
        """
        return np.random.normal(loc= 0 ,scale=0.01,size = shape)

    def feedforward(self,X):
        
        pre_activation_val = {}
        activation_val     = {}
        # 'relu', 'sigmoid', 'linear', 'tanh', 'softmax'

        layer_inp = X
        activation_val[0] = layer_inp
        for i in range(1,self.n_layers -1):
            pre_activation_val[i] = np.dot(layer_inp,self.weight[i]) + self.bias[i]
        
            if(self.activation == "sigmoid"):
                activation_val[i]     = self.sigmoid(pre_activation_val[i])  

            if(self.activation == "tanh"):                
                activation_val[i]     = self.tanh(pre_activation_val[i])  

            if(self.activation == "relu"):
                activation_val[i]     = self.relu(pre_activation_val[i])  

            if(self.activation == "linear"):
                activation_val[i]     = self.linear(pre_activation_val[i])  
                    
            layer_inp = activation_val[i]      
                    
        pre_activation_val[self.n_layers-1] = np.dot(layer_inp,self.weight[self.n_layers-1]) + self.bias[self.n_layers-1]                    
        activation_val[self.n_layers-1]     = self.softmax(pre_activation_val[self.n_layers-1])
        return (pre_activation_val,activation_val)

    
    def backpropagation(self,y_orig,pre_activation_val,activation_val):
        gradients = {}
        cur_layer = self.n_layers - 1
        gradients[cur_layer] = activation_val[cur_layer] - y_orig
        der_activation = {}
        for cur_layer in range(self.n_layers - 2,0,-1):
            der_activation[cur_layer] = np.dot(gradients[cur_layer+1],self.weight[cur_layer+1].T)            

            if(self.activation == "sigmoid"):
                gradients[cur_layer]    = der_activation[cur_layer]*self.sigmoid_grad(pre_activation_val[cur_layer])
            elif(self.activation == "tanh"):
                gradients[cur_layer]    = der_activation[cur_layer]*self.tanh_grad(pre_activation_val[cur_layer])
            elif(self.activation == "relu"):
                gradients[cur_layer]    = der_activation[cur_layer]*self.relu_grad(pre_activation_val[cur_layer])
            elif(self.activation == "linear"):
                gradients[cur_layer]    = der_activation[cur_layer]*self.linear_grad(pre_activation_val[cur_layer])

        for i in range(self.n_layers-1,0,-1):
            self.weight[i] -= (self.learning_rate/self.batch_size) * np.dot(activation_val[i-1].T,gradients[i])
            self.bias[i] -= (self.learning_rate/self.batch_size)* np.sum(gradients[i],axis=0,keepdims=True)

        return gradients

    def one_hot(self,y):
        n_samples = y.size
        n_labels  = y.max() + 1
        y_one_hot = np.zeros((n_samples,n_labels))
        # below line y[row,col] = 1
        #fancy indexing numpy
        y_one_hot[np.arange(n_samples),y] = 1
        return y_one_hot

    def get_mini_batch(self,X,y):
        mini_batches = []
        x_y_combined = np.hstack((X,y))
        np.random.shuffle(x_y_combined)
        X = x_y_combined[:,:X.shape[1]]
        y = x_y_combined[:,X.shape[1]:]
        num_min_batch = X.shape[0]//self.batch_size

        i=0
        while i < num_min_batch:
            X_mini = X[i * self.batch_size:(i + 1)*self.batch_size, :]
            y_mini = y[i * self.batch_size:(i + 1)*self.batch_size, :]
            mini_batches.append([X_mini,y_mini])
            i+=1
        if(i*self.batch_size != X.shape[0]):
            X_mini = X[i * self.batch_size :, :]
            y_mini = y[i * self.batch_size :, :]
            mini_batches.append([X_mini,y_mini])

        return mini_batches

    def fit(self, X, y,X_valid=None, y_valid=None):
        """
        Fitting (training) the linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as training data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as training labels.
        
        X_valid : 2-dimensional numpy array of shape (n_samples, n_features) which acts as validation data.

        y_valid : 1-dimensional numpy array of shape (n_samples,) which acts as validation labels.

        Returns
        -------
        self : an instance of self
        """

        # fit function has to return an instance of itself or else it won't work with test.py
        y_orig = self.one_hot(y)
        if y_valid is not None:
            y_valid_orig = self.one_hot(y_valid)
        n_samples  = X.shape[0]
        inp_layer_size = X.shape[1]

        mini_batches = self.get_mini_batch(X,y_orig)
        self.training_loss = []
        self.validation_loss = []
        
        for iter_ in (range(self.num_epochs)):
            for batch in mini_batches:
                cur_batch_x = batch[0]
                cur_batch_y = batch[1]
                cur_samples = cur_batch_x.shape[0]
                loss = []
                cur_val_loss = []
                pre_activation_val,activation_val = self.feedforward(cur_batch_x)
                loss.append(self.cross_entropy_loss(cur_batch_y,activation_val[self.n_layers-1]))
                self.backpropagation(cur_batch_y,pre_activation_val,activation_val)
            print("Iteration :",iter_," Loss :",np.array(loss).mean())
            
            self.training_loss.append(np.array(loss).mean())
            
            if X_valid is not None:
                y_pred_val = self.predict_proba(X_valid)
                self.validation_loss.append(self.cross_entropy_loss(y_valid_orig,y_pred_val))
                    
        return self


    def cross_entropy_loss(self,y_orig,y_pred):
        #fancy indexing numpy 
        cross_entropy = np.sum(-np.log(y_pred[np.arange(y_orig.shape[0]),y_orig.argmax(axis=1)]))
        return cross_entropy/y_orig.shape[0]

    def predict_proba(self, X):
        """
        Predicting probabilities using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        Returns
        -------
        y : 2-dimensional numpy array of shape (n_samples, n_classes) which contains the 
            class wise prediction probabilities.
        """
        
        pre_activation_val,activation_val = self.feedforward(X)

        return activation_val[self.n_layers-1]

    def predict(self, X):
        """
        Predicting values using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        Returns
        -------
        y : 1-dimensional numpy array of shape (n_samples,) which contains the predicted values.
        """

        # return the numpy array y which contains the predicted values
        probab = self.predict_proba(X)
        return probab.argmax(axis=1)

    def score(self, X, y):
        """
        Predicting values using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as testing labels.

        Returns
        -------
        acc : float value specifying the accuracy of the model on the provided testing set
        """
        # return the numpy array y which contains the predicted values
        y_pred = self.predict(X)
        count=0
        for i in range(len(y_pred)):
            if(y_pred[i]==y[i]):
                count+=1
        return count/len(y_pred)
    
    def save(self,path_with_name):
        model_op = open(path_with_name,"wb")
        pickle.dump(self,model_op)
        model_op.close()