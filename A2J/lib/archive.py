def test():

    net = model.A2J_model(num_classes = keypointsNumber, is_3D=is_3D)
    net.load_state_dict(torch.load(os.path.join(check_dir, 'epoch#25_lr_0.00035_wetD_0.00010_stepSize_10_gamma_0.pth')))
    net = net.cuda()
    net.eval()
    
    post_process = anchor.post_process(is_3D=is_3D)

    output = torch.FloatTensor()

    while True:
        for img, mean, box in tqdm(test_dataloader):
            with torch.no_grad():
                img, mean, box = img.cuda(), mean.cuda(), box.cuda()
                heads = net(img)  
                pred_keypoints = post_process(heads, mean, box).data.cpu()
                output = torch.cat([output, pred_keypoints], 0)

        result = output.numpy()

        for i in range(result.shape[0]-1):
            result[i+1] = (result[i] + result[i+1]) / 2

        np.save(os.path.join(res_dir, f'{data_name}_result.npy'), result)

def pixel2world(pixel_coord, scale=1, width=160.0, height=120.0, fx=0.0035, fy=0.0035):
    world_coord = pixel_coord.copy()
    
    x = world_coord[:,:,0]
    y = world_coord[:,:,1]
    z = world_coord[:,:,2]
 
    world_coord[:,:,0] = (x / scale - width) * fx * z
    world_coord[:,:,1] = (height - y / scale) * fy * z
    world_coord[:,:,2] = z
    
    return world_coord