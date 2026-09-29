!> Trace relativistic guiding centres through the fields of an existing JOREK run.
!>
!> A configurable version of ex7_jorek, for ashen's `trace` (a case's
!> trace_exe pointing at the built binary, trace_params.nml given in its
!> trace_inputs). It has never been compiled -- expect a first build to need
!> small fixes. Everything ex7_jorek hard-codes is read from trace_params.nml
!> (example: ashen's fortran/trace_params.example.nml):
!>
!>   &trace
!>     field_mode    = 'evolving'  ! 'static': jorek_restart.h5 only, frozen
!>                                 ! 'evolving': jorek<i>.h5, jorek<i+1>.h5, ...
!>                                 !   interpolated linearly in time
!>     restart_index = 0           ! evolving: index of the first restart file
!>     hold_last_field = .false.   ! evolving: past the last restart, keep its
!>                                 !   field frozen (.true.) or stop (.false.)
!>     t_span        = 1.d-5       ! [s] traced, from the first restart's time
!>     dt            = 1.d-10      ! [s] RK4 step
!>     diag_step     = 1.d-8       ! [s] between diagnostics writes
!>     snapshot_step = 1.d-6       ! [s] between part_restart<time>.h5 files;
!>                                 !   0 = only the final part_restart.h5
!>     mass          = 5.48579909065d-4  ! [amu]
!>     n_markers     = 1
!>     R0 = 3.68  Z0 = 0.  phi0 = 0.     ! [m], [m], [rad]
!>     E_kin_eV = 1.d7                   ! kinetic energy [eV]
!>     cos_pitch = 0.                    ! v_par/v
!>     charge = -1                       ! [e]
!>   /
!>
!> Under ashen, keep restart_index = 0: ashen links the chosen restarts in
!> as a consecutive sequence from index 0 (the reader looks only i+1..i+20
!> past file i, at rst_file_ind_fmt(1)'s width), and links jorek_restart.h5
!> to the start restart for field_mode = 'static'. See ashen.tracing.
!>
!> The start time is the first restart's own time, never set separately:
!> the field reader sets sim%time from the file, and a separately given
!> start time that disagreed with it would only be warned about.
!>
!> Differences from ex7_jorek:
!> - a particle that leaves the grid is flagged lost (i_elm <= 0) and no
!>   longer pushed; the run carries on with the others instead of `stop`
!> - each particle is pushed with its own local time (re_gc_current_density_
!>   initialisation's loop). ex7 advanced sim%time inside the particle loop,
!>   so with more than one particle every later one saw fields at the wrong time
!> - markers are shared round-robin between MPI ranks
!>
!> Outputs: trace_diag.h5 (write_particle_diagnostics: energy, mu, psi_n,
!> p_phi, lost, phi, R, Z per diag_step), part_restart<time>.h5 every
!> snapshot_step and part_restart.h5 at the end (write_simulation_hdf5, the
!> full particle state) -- the files `plot --diag particles` draws.
!>
!> Kept in ashen, not in JOREK (which ashen never modifies). To build, copy
!> it into a JOREK checkout's particles/examples/ -- the Makefile picks up
!> any program there by filename -- then `make trace_gc` with the same MODEL
!> as the run being traced.
!> Run with `mpirun -n N ./trace_gc < in_main`, next to trace_params.nml.

program trace_gc

! Every import is named: particle_tracer re-exports everything it uses
! (phys_module's t_start among it), and a local name clashing with any of
! that would not compile. find_RZ is the external grids/grid_utils/find_RZ.f90.
use particle_tracer,          only: sim, events, particle_sim, particle_gc_relativistic, &
                                    event, with, next_event_at, stop_action
use mod_particle_io,          only: write_simulation_hdf5
use mod_io_actions,           only: write_action
use mod_particle_diagnostics, only: write_particle_diagnostics
use mod_fields_linear,        only: read_jorek_fields_interp_linear
use mod_gc_relativistic,      only: runge_kutta_fixed_dt_gc_push_jorek, &
                                    relativistic_gc_momenta_from_E_cospitch
use constants,                only: ATOMIC_MASS_UNIT, EL_CHG, SPEED_OF_LIGHT
use mpi

implicit none

integer, parameter :: MAX_MARKERS = 10000
character(len=*), parameter :: PARAMS_FILE = 'trace_params.nml'

! --- &trace namelist ---
character(len=16) :: field_mode = 'evolving'
integer           :: restart_index = 0
logical           :: hold_last_field = .false.
real*8            :: t_span = 1.d-5, dt = 1.d-10, diag_step = 1.d-8, snapshot_step = 0.d0
real*8            :: mass = 5.48579909065d-4
integer           :: n_markers = 0
real*8            :: R0(MAX_MARKERS) = 0.d0, Z0(MAX_MARKERS) = 0.d0, phi0(MAX_MARKERS) = 0.d0
real*8            :: E_kin_eV(MAX_MARKERS) = 0.d0, cos_pitch(MAX_MARKERS) = 0.d0
integer           :: charge(MAX_MARKERS) = -1
namelist /trace/ field_mode, restart_index, hold_last_field, t_span, dt, diag_step, &
                 snapshot_step, mass, n_markers, R0, Z0, phi0, E_kin_eV, cos_pitch, charge

! sim and events come from particle_tracer
type(event)                      :: field_reader
type(write_particle_diagnostics) :: diag
type(write_action)               :: snapshots
type(particle_gc_relativistic)   :: marker
real*8                           :: t_start, target_time, rest_energy_eV
integer                          :: u, io, ierr, k, j, n_local, ifail, n_lost
integer                          :: my_rank, n_ranks

call sim%initialize(num_groups=1)
! Asked of MPI directly: particle_sim's own rank fields differ between JOREK versions.
call MPI_COMM_RANK(MPI_COMM_WORLD, my_rank, ierr)
call MPI_COMM_SIZE(MPI_COMM_WORLD, n_ranks, ierr)

! --- parameters: read on rank 0, broadcast ---
if (my_rank .eq. 0) then
  open(newunit=u, file=PARAMS_FILE, status='old', action='read', iostat=io)
  if (io .ne. 0) then
    write(*,*) 'ERROR: trace_gc: cannot open ', PARAMS_FILE
    call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
  end if
  read(u, nml=trace, iostat=io)
  close(u)
  if (io .ne. 0) then
    write(*,*) 'ERROR: trace_gc: cannot read the &trace namelist in ', PARAMS_FILE
    call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
  end if
  if (n_markers .lt. 1 .or. n_markers .gt. MAX_MARKERS) then
    write(*,*) 'ERROR: trace_gc: n_markers must be in 1..', MAX_MARKERS, ', got ', n_markers
    call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
  end if
  if (snapshot_step .lt. 0.d0) then
    write(*,*) 'ERROR: trace_gc: snapshot_step must be >= 0, got ', snapshot_step
    call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
  end if
  if (trim(field_mode) .ne. 'static' .and. trim(field_mode) .ne. 'evolving') then
    write(*,*) "ERROR: trace_gc: field_mode must be 'static' or 'evolving', got ", trim(field_mode)
    call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
  end if
end if
call MPI_Bcast(field_mode, len(field_mode), MPI_CHARACTER, 0, MPI_COMM_WORLD, ierr)
call MPI_Bcast(restart_index, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)
call MPI_Bcast(hold_last_field, 1, MPI_LOGICAL, 0, MPI_COMM_WORLD, ierr)
call MPI_Bcast(t_span, 1, MPI_REAL8, 0, MPI_COMM_WORLD, ierr)
call MPI_Bcast(dt, 1, MPI_REAL8, 0, MPI_COMM_WORLD, ierr)
call MPI_Bcast(diag_step, 1, MPI_REAL8, 0, MPI_COMM_WORLD, ierr)
call MPI_Bcast(snapshot_step, 1, MPI_REAL8, 0, MPI_COMM_WORLD, ierr)
call MPI_Bcast(mass, 1, MPI_REAL8, 0, MPI_COMM_WORLD, ierr)
call MPI_Bcast(n_markers, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)
call MPI_Bcast(R0, n_markers, MPI_REAL8, 0, MPI_COMM_WORLD, ierr)
call MPI_Bcast(Z0, n_markers, MPI_REAL8, 0, MPI_COMM_WORLD, ierr)
call MPI_Bcast(phi0, n_markers, MPI_REAL8, 0, MPI_COMM_WORLD, ierr)
call MPI_Bcast(E_kin_eV, n_markers, MPI_REAL8, 0, MPI_COMM_WORLD, ierr)
call MPI_Bcast(cos_pitch, n_markers, MPI_REAL8, 0, MPI_COMM_WORLD, ierr)
call MPI_Bcast(charge, n_markers, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)

! --- fields: reading them sets sim%time to the first restart's time ---
if (trim(field_mode) .eq. 'static') then
  field_reader = event(read_jorek_fields_interp_linear(i=-1))
else
  field_reader = event(read_jorek_fields_interp_linear(i=restart_index, &
                                                       stop_at_end=.not. hold_last_field))
end if
call with(sim, field_reader)
t_start = sim%time
if (my_rank .eq. 0) write(*,'(A,ES14.6,A)') 'trace_gc: start time ', t_start, ' s'

! --- markers, round-robin over ranks ---
sim%groups(1)%mass = mass
n_local = 0
do k = 1, n_markers
  if (mod(k-1, n_ranks) .eq. my_rank) n_local = n_local + 1
end do
allocate(particle_gc_relativistic::sim%groups(1)%particles(n_local))

rest_energy_eV = mass * ATOMIC_MASS_UNIT * SPEED_OF_LIGHT**2 / EL_CHG
j = 0
do k = 1, n_markers
  if (mod(k-1, n_ranks) .ne. my_rank) cycle
  j = j + 1
  select type (p => sim%groups(1)%particles(j))
  type is (particle_gc_relativistic)
    p%q      = int(charge(k), kind=1)
    p%x      = [R0(k), Z0(k), phi0(k)]
    call find_RZ(sim%fields%node_list, sim%fields%element_list, &
                 R0(k), Z0(k), &                                        ! inputs
                 p%x(1), p%x(2), p%i_elm, p%st(1), p%st(2), ifail)      ! outputs
    if (p%i_elm .le. 0) then
      write(*,'(A,I0,A,2F10.5,A)') 'WARNING: trace_gc: marker ', k, ' at (R,Z) = (', &
        R0(k), Z0(k), ') is outside the grid; it is reported lost from the start'
      p%i_elm = 0
      cycle
    end if
    if (abs(cos_pitch(k)) .gt. 1.d0) then
      write(*,'(A,I0,A,ES12.4)') 'ERROR: trace_gc: marker ', k, ' has |cos_pitch| > 1: ', cos_pitch(k)
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    end if
    marker = p
    marker = relativistic_gc_momenta_from_E_cospitch(marker, E_kin_eV(k) + rest_energy_eV, &
                                                     cos_pitch(k), mass, sim%fields, sim%time)
    p%p = marker%p
  end select
end do

! --- events: diagnostics (and snapshots) from the start, stop t_span later ---
diag = write_particle_diagnostics(filename='trace_diag.h5', &
                                  only=[1,2,3,4,6,8,12,13,14,15]) ! e, k, mu, psi_n, p_phi, lost, phi, R, Z, i_elm
if (snapshot_step .gt. 0.d0) then
  ! part_restart<time>.h5, named from the time as re_gc's snapshots are
  snapshots = write_action(basename='part_restart')
  events = [field_reader, &
            event(diag, start=t_start, step=diag_step), &
            event(snapshots, start=t_start, step=snapshot_step), &
            event(stop_action(), start=t_start + t_span)]
else
  events = [field_reader, &
            event(diag, start=t_start, step=diag_step), &
            event(stop_action(), start=t_start + t_span)]
end if
call with(sim, events, at=sim%time)

! --- push ---
do while (.not. sim%stop_now)
  target_time = next_event_at(sim, events)
  call push_all(sim, dt, target_time)
  sim%time = target_time
  call with(sim, events, at=sim%time)
end do

n_lost = 0
select type (particles => sim%groups(1)%particles)
type is (particle_gc_relativistic)
  n_lost = count(particles(:)%i_elm .le. 0)
end select
call MPI_Allreduce(MPI_IN_PLACE, n_lost, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD, ierr)
if (my_rank .eq. 0) write(*,'(A,I0,A,I0,A)') 'trace_gc: done, ', n_lost, ' of ', &
  n_markers, ' markers lost'

call write_simulation_hdf5(sim, 'part_restart.h5')
call sim%finalize

contains

!> Push every marker from sim%time to target_time with fixed-step RK4.
!> A lost marker (i_elm <= 0) is left where it left the grid.
subroutine push_all(sim, dt, target_time)
  type(particle_sim), intent(inout) :: sim
  real*8, intent(in)                :: dt, target_time
  integer :: i
  real*8  :: local_time, local_dt
  !$omp parallel do default(none) firstprivate(dt, target_time) &
  !$omp private(i, local_time, local_dt) shared(sim)
  do i = 1, size(sim%groups(1)%particles)
    select type (gc => sim%groups(1)%particles(i))
    type is (particle_gc_relativistic)
      local_time = sim%time
      local_dt   = min(dt, target_time - local_time)
      do while ((target_time - local_time .gt. 0.d0) .and. (gc%i_elm .gt. 0))
        call runge_kutta_fixed_dt_gc_push_jorek(sim%fields, local_time, local_dt, &
                                                sim%groups(1)%mass, gc)
        local_time = local_time + local_dt
        local_dt   = min(dt, target_time - local_time)
      end do
    end select
  end do
  !$omp end parallel do
end subroutine push_all

end program trace_gc
