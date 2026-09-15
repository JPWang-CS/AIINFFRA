// Key excerpts from NVIDIA CUTLASS v4.6.3:
// examples/cute/tutorial/blackwell/03_mma_tma_multicast_sm100.cu
// This is a source-check fixture, not a standalone translation unit.
uint16_t tma_mcast_mask_a = create_tma_multicast_mask<2>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk);
uint16_t tma_mcast_mask_b = create_tma_multicast_mask<1>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk);
uint16_t mma_mcast_mask_c = create_tma_multicast_mask<0,1>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk) |
                            create_tma_multicast_mask<0,2>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk);

int num_mcast_participants = size<1>(cluster_layout_vmnk) + size<2>(cluster_layout_vmnk) - 1;
cute::initialize_barrier(shared_storage.mma_barrier, /* num_ctas */ num_mcast_participants);
cute::initialize_barrier(shared_storage.tma_barrier, /* num_threads */ 1);

cute::set_barrier_transaction_bytes(shared_storage.tma_barrier, tma_transaction_bytes);
copy(tma_atom_A.with(shared_storage.tma_barrier, tma_mcast_mask_a), tAgA(_,k_tile), tAsA);
copy(tma_atom_B.with(shared_storage.tma_barrier, tma_mcast_mask_b), tBgB(_,k_tile), tBsB);
cute::wait_barrier(shared_storage.tma_barrier, tma_barrier_phase_bit);

cutlass::arch::umma_arrive_multicast(&shared_storage.mma_barrier, mma_mcast_mask_c);
cute::wait_barrier(shared_storage.mma_barrier, mma_barrier_phase_bit);
mma_barrier_phase_bit ^= 1;
