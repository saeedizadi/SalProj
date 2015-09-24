function [curr_pos] = store2hdf5(filename, data,label,create,chunksz,curr_pos)

data = reshape(data,[1 16896 1 size(data,1)]);
startloc=struct('Data',[1,1,1,curr_pos+1],'LB',[1,curr_pos+1]);

if create
    startloc=struct('Data',[1,1,1,1],'LB',[1,1]);
    h5create(filename, '/data', [1 16896 1 Inf], 'Datatype', 'double', 'ChunkSize', [1 16896 1 chunksz]);
    h5create(filename, '/label', [1 Inf], 'Datatype', 'double', 'ChunkSize', [1 chunksz]); % width, height, channels, number
    
end
h5write(filename, '/data', data, startloc.Data,size(data));
h5write(filename, '/label', label, startloc.LB, size(label));
curr_pos = size(data,4);

end
