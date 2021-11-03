# tiny-transformer
tiny transformer(s) on pytorch

## Why transformers ? 

The answer lies in the disadvantages we faced in the RNNs and the vanilla convnets:

* **RNNs**: For a given input sequence, we cannot compute the output for time step `i` until we've calculated the output for `i-1`.
* **ConvNets**: Only the words which are closer (i.e within the kernel size) together are able to interact which each other. This hinders long range interactions. 

The transformer is a solution to both the issues, it enables us to map long range interactions without any recurrent connections. 


## Resources
- [Youtube video that explains transformers](https://www.youtube.com/watch?v=U0s0f995w14)
- [Blog post by peterbloem](http://peterbloem.nl/blog/transformers)