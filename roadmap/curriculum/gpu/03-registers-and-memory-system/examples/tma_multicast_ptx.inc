// PTX ISA tensor-copy multicast instruction; operands are illustrative.
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster
    [smem_dst], [tensor_map, {tile_x, tile_y}], [tma_bar], cta_mask;
