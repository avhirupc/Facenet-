import tensorflow as tf 
from Utils import utilitites,ops
import model
import numpy
import numpy as np
import random

def train(learning_rate=0.01,alpha=0.01):
    images_placeholder,embeddings=model.NN1(batch_size=90).build_model()

    batches=utilitites.get_minibatches('data/train')

    anchor, positive, negative = tf.unstack(tf.reshape(embeddings, [-1,3,128]), 3, 1)
    triplet_loss = ops.triplet_loss(anchor, positive, negative, alpha)
       
    t_vars=tf.trainable_variables()
    train_op=tf.train.AdamOptimizer(learning_rate, beta1 = 0.01).minimize(triplet_loss, var_list=t_vars)
    
    #saving model
    saver=tf.train.Saver()

    init1=tf.global_variables_initializer()
    init2=tf.local_variables_initializer()
    with tf.Session() as sess:
        sess.run(init1)
        sess.run(init2)
        number_of_epochs=1
        for epoch in range(number_of_epochs):
            i=0
            for batch in batches:
                try:
                    images,images_paths=utilitites.get_batch_to_numpy(batch)
                    #find embeddings 
                    print "Finding Embeddings for Images"
                    emb=sess.run(embeddings,feed_dict={images_placeholder:images})
                    #using these embeddings find pairs to select
                    #select triplet function return image_path of triplets
                    #triplet_image_acc_to_embeddings=select_triplets(emb,images_paths)
                    print "Selecting Triplets"
                    select_batch_for_loss=select_triplets(emb,images_paths)
                    triplets,triplets_path=utilitites.get_batch_to_numpy(select_batch_for_loss)
                    #print triplets.shape
                    #make a batch using these triplets
                    #pass it as images batch to minimize loss
                    i+=1
                    print "Calculating Loss"
                    _,loss=sess.run([train_op,triplet_loss],feed_dict={images_placeholder:triplets})
                    print("loss:",loss)
                    TA,FA,same,diff=calculate_true_accepts_false_accepts(emb,images_paths)
                    VAL=float(TA)/float(same)
                    FAR=float(FA)/float(diff)
                    print("TrueAccepts: %d,FalseAccepts: %d Number Of Same Pairs:%d Number Of different Pairs:%d VAL:%f FAR:%f" %(TA,FA,same,diff,VAL,FAR))
                    #saver.save(sess, "latest_model.ckpt")
                except:
                    print "Batch Skipped Due to IO Error"


def select_triplets(emb,images_paths,alpha=0.1,batch_size=90):
    """
    Rearranges image_paths so that they when unstacked form proper pair of triplets
    Args:
    emb :embedding of all the iamges in current batch
    image_paths : batch of image paths
    alpha : threshold hyperparameter
    batch_size :batch_size
    
    Returns:
    Rearranged image_path forming proper triplets
    """
    new_image_paths_order=[]
    images_dict={}
    i=0
    for image_path in images_paths:
        label=image_path[11:][:-69]
        if label in images_dict.keys():
            images_dict[label].append({image_path:i})
        else:
            images_dict[label]=[{image_path:i}]
        i+=1     
    identities_in_batch=images_dict.keys()
    i=0
    j=0
    leftout_images=[]
    for z in range(len(images_paths)/3):
        current_identity=identities_in_batch[i]
        
        if(len(images_dict[current_identity])<=1):
            i+=1
            current_identity=identities_in_batch[i]
            

        anchor=images_dict[current_identity][0]         #always selecting first and second element and removig them from list after computation
        positive=images_dict[current_identity][1]
        #Removing 
        images_dict[current_identity].pop(0)
        images_dict[current_identity].pop(0)

        anchor_embd_ind=anchor.values()[0]
        positive_embd_ind=positive.values()[0]
        pos_dist_sqr=np.sum(np.square(emb[anchor_embd_ind]-emb[positive_embd_ind]))
            
        options_for_neg=[]
        for k in range(i+1,len(identities_in_batch)):
           options_for_neg+=images_dict[identities_in_batch[k]]
        
        temp_indeces=range(len(options_for_neg))
        random.shuffle(temp_indeces)
        hard_neg_not_found=True
        negative=None

        while hard_neg_not_found:
                for ind in temp_indeces:
                    possible_neg=options_for_neg[ind]
                    possible_neg_ind=possible_neg.values()[0]
                    neg_dist_sqr=np.sum(np.square(emb[anchor_embd_ind]-emb[possible_neg_ind]))
                    if(pos_dist_sqr- neg_dist_sqr< alpha):
                        negative=possible_neg
                        hard_neg_not_found=False
                        break


    
        new_image_paths_order.append(anchor.keys()[0])
        new_image_paths_order.append(positive.keys()[0])
        new_image_paths_order.append(negative.keys()[0])
        
    return new_image_paths_order

def calculate_true_accepts_false_accepts(embeddings,images_paths,threshold=0.1e-06,batch_size=90):
    TA=0
    FA=0
    psame=0
    pdiff=0
    for i in range(batch_size):
        for j in range(1,batch_size):
            emb1=embeddings[i]
            emb2=embeddings[j]
            diff = np.subtract(emb1, emb2)[0]
            dist = np.sum(np.square(diff),1)[0]
            if np.less(dist,threshold):
                if images_paths[i][11:][:-69]==images_paths[j][11:][:-69]:
                    TA+=1
                else:
                    FA+=1
            if images_paths[i][11:][:-69]==images_paths[j][11:][:-69]:
                psame+=1
            else:
                pdiff+=1
    return TA,FA,psame,pdiff


if __name__ == '__main__':
    train()
    