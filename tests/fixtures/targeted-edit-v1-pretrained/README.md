# Historical v1 reconstruction source

`pretrained.py` is the exact source published at commit
`3581c974cbaf20d3118bd825a8c9c68188774b10`, with SHA-256
`ef86be0af1fd548027d23c8de779ff8cc77a8e180a9cb05d0b10a919454752da`.
The published v1 frozen manifest records that digest. The v2 synthetic Git
fixtures copy these bytes into their isolated repositories so that their source
commits match the unchanged historical evidence. They do not import or execute
this copy. The fixture-local attributes preserve its LF bytes on checkout.
